---
layout: post
title: "Provisioning AWS infrastructure for deep learning project"
date: 2024-10-28 00:00:00 -0000
categories: 
    - cloud
    - terraform
tags: ["cloud", "aws", "terraform", "shell"]
---

# Provisioning AWS infrastructure for deep learning project

You could say this article is part 2 of [this one](https://mmalek06.github.io/cloud/2024/10/27/provisioning-azure-infrastructure-for-deep-learning-project.html). In that post, I described Azure infrastructure provisioning for my deep learning model training and noted that my original idea was to use AWS infrastructure. However, I had to submit a quota increase request and ended up waiting almost two days for AWS support to process it. Eventually, the request was approved, so now I can give AWS a try.

## Requirements

- Provision a VM on which various deep learning libraries can be installed
- Enable RDP on it (only from my IP)
- Mount AWS S3 on it

## The code

The providers section is not much different from the one in the previous post, with one exception—in that setup, I specified the subscription ID, while here I'm setting the profile and region that I'll use for this project.

```plaintext
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~>3.0"
    }
  }
}

provider "aws" {
  region  = "eu-central-1"
  profile = "terraformdeeplearning"
}

provider "http" {}
```

Next comes the security group definition, along with a surprise (to me). It turns out that Azure's approach of mandating virtual networks for VMs is not required in AWS. AWS takes a "we'll give you some sensible defaults" approach, meaning no advanced networking setup is necessary.

Of course, in production-ready code, it wouldn't be this simple, but my requirement here is straightforward: a single, RDP-capable VM in the cloud, accessible only from my IP address. This is as simple as the code gets:

```plaintext
data "http" "my_ip" {
  url = "https://checkip.amazonaws.com/"
}

locals {
  my_ip_cidr = "${chomp(data.http.my_ip.response_body)}/32"
}

resource "aws_security_group" "rdp_access" {
  name        = "rdp_access_sg"
  description = "Allow RDP access only from my IP"

  ingress {
    from_port   = 3389
    to_port     = 3389
    protocol    = "tcp"
    cidr_blocks = [local.my_ip_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

Now, onto security and enabling S3 access from my EC2 instance. The `aws_s3_bucket_policy` specifies that the role used by the EC2 instance will have access to the bucket. This is complemented by the `aws_iam_policy`, which grants the role itself permissions to interact with S3.

In other words, the first policy gives explicit access to the specified bucket - without it, even if the EC2 role had permissions to perform S3 actions, requests would be denied. The second policy grants the necessary S3 permissions. It's like a lock with two keys.

```plaintext
resource "aws_s3_bucket" "deep_learning_bucket" {
  bucket = "deep-learning-bucket-mm"

  tags = {
    Name = "Deep learning data bucket"
  }
}

resource "aws_s3_bucket_policy" "deep_learning_bucket_policy" {
  bucket = aws_s3_bucket.deep_learning_bucket.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          "AWS" : "${aws_iam_role.ec2_role.arn}"
        },
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ],
        Resource = [
          "${aws_s3_bucket.deep_learning_bucket.arn}",
          "${aws_s3_bucket.deep_learning_bucket.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_policy" "s3_access_policy" {
  name        = "S3AccessPolicy"
  description = "Policy to allow EC2 access to a specific S3 bucket"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
        Resource = [
          aws_s3_bucket.deep_learning_bucket.arn,
          "${aws_s3_bucket.deep_learning_bucket.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role" "ec2_role" {
  name = "EC2S3AccessRole"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "s3_policy_attachment" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.s3_access_policy.arn
}

resource "aws_iam_instance_profile" "ec2_instance_profile" {
  name = "EC2InstanceProfile"
  role = aws_iam_role.ec2_role.name
}
```

As for the VM definition itself - it's also simpler than the corresponding one on Azure. For example there's no need to provision public IP explicitly, network interface and other network related resources:

```plaintext
resource "random_password" "password" {
  length      = 10
  min_lower   = 1
  min_upper   = 1
  min_numeric = 1
  min_special = 1
  special     = true
}

resource "aws_instance" "g5xlarge_instance" {
  ami                  = "ami-0084a47cc718c111a"
  instance_type        = "g5.xlarge"
  iam_instance_profile = aws_iam_instance_profile.ec2_instance_profile.name

  user_data = <<-EOF
            #!/bin/bash
            apt-get update -y
            apt-get install -y ubuntu-desktop xrdp s3fs
            
            systemctl enable xrdp
            systemctl start xrdp

            mkdir -p /mnt/${aws_s3_bucket.deep_learning_bucket.bucket}
            s3fs ${aws_s3_bucket.deep_learning_bucket.bucket} /mnt/${aws_s3_bucket.deep_learning_bucket.bucket} -o iam_role=auto
            echo "s3fs#${aws_s3_bucket.deep_learning_bucket.bucket} /mnt/${aws_s3_bucket.deep_learning_bucket.bucket} fuse _netdev,iam_role=auto,allow_other 0 0" >> /etc/fstab
            
            pip3 install --upgrade pip
            pip3 install opencv-python-headless matplotlib numpy torch torchvision tqdm jupyter
            mkdir -p $HOME/jupyter_notebooks
            sudo chown -R ubuntu $HOME/jupyter_notebooks/
            jupyter notebook --generate-config -y
            echo "c.NotebookApp.ip = '0.0.0.0'" >> $HOME/.jupyter/jupyter_notebook_config.py
            echo "c.NotebookApp.open_browser = False" >> $HOME/.jupyter/jupyter_notebook_config.py
            echo "c.NotebookApp.notebook_dir = '$HOME/jupyter_notebooks'" >> $HOME/.jupyter/jupyter_notebook_config.py
            echo "c.NotebookApp.allow_root = True" >> $HOME/.jupyter/jupyter_notebook_config.py

            echo "ubuntu:${random_password.password.result}" | chpasswd
            EOF


  vpc_security_group_ids = [aws_security_group.rdp_access.id]

  tags = {
    Name = "Ubuntu_Pro_G5_Instance_with_RDP"
  }
}
```

## Summary

As a Microsoft/Azure fanboy I had a hard pill to swallow - after the initial hickup, I actually enjoyed working with AWS more because it only required the most esential definitions to be explicitly created. 
