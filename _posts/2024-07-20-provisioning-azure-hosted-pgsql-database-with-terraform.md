---
layout: post
title: "Provisioning an Azure-hosted PostgreSQL database with Terraform"
date: 2024-07-19 15:00:00 -0000
categories: Cloud
tags: ["cloud", "azure", "terraform", "c#", "shell", "powershell"]
---

# Provisioning an Azure-hosted PostgreSQL database with Terraform

A while ago, while working on a passion project, I grew tired of juggling Docker containers locally. With only 16GB of RAM, I felt the need to find a solution that could alleviate my machine's resource pressure and free me from constantly turning Docker containers on and off, depending on the project. I usually use Azure cloud for my personal projects, partly because I'm a huge Microsoft fanboy and find configuring Azure easier.

On more than one occasion, I've found AWS documentation slightly out-of-date, with various settings renamed in the cloud interface but not updated in the docs. In my opinion, Microsoft handles this better. While the use of Terraform can alleviate this problem, every time I get blocked, I prefer having up-to-date documentation to inform my next steps and help me understand what Terraform is actually doing.

---

## Requirements

There was only a single requirement that I had in mind - I wanted to make the database visible only from my machine for the time being. Using Azure VPN would not be a good option for this scenario, because it was just a hobby project. 

## The code

First some helper scripts. Terraform will need to know the current machine IP in order to define a firewall rule.

```shell
#!/bin/bash
IP=$(curl -s https://ifconfig.me)

echo "{\"ip\":\"$IP\"}"

```

```powershell
$IP = Invoke-RestMethod -Uri 'https://ifconfig.me'

Write-Output "{`"ip`":`"$IP`"}"
```

And now the datasourcing Terraform code:

```plaintext
locals {
  is_linux = length(regexall("/home/", lower(abspath(path.root)))) > 0
}

data "external" "public_ip" {
  program = local.is_linux ? ["${path.module}/../../scripts/get-ip.sh"] : ["powershell", "-File", "${path.module}/../../scripts/GetIp.ps1"]
}
```

The first declaration checks if the current platform is Linux-based. I needed this information during the data sourcing step to run the correct script for getting the IP. There are many ways to check the OS in Terraform, but the shortest one is probably to check for the existence of the /home directory. After that, it’s time to create the DB declaration:

```plaintext
resource "azurerm_postgresql_flexible_server" "passion_project_db" {
  name                          = "passion-pgsql-db"
  location                      = var.resource_group_location
  resource_group_name           = var.resource_group_name
  version                       = "13"
  administrator_login           = "passionadmin"
  administrator_password        = var.pg_password
  sku_name                      = "B_Standard_B1ms"
  storage_mb                    = 32768
  backup_retention_days         = 7
  delegated_subnet_id           = null

  tags = {
    environment = "${var.environment_name}"
  }
}

```

The above works, but if the PostgreSQL Flexible Server fails-over to the Standby Availability Zone, the zone will be updated to reflect the current Primary Availability Zone. The thing with Terraform is that a lot happens under the hood. The above declaration doesn’t explicitly set the availability zone, meaning some default value will be used. If a fail-over occurs, Terraform will detect a change and decide to redeploy the resource, resulting in two databases being created. To prevent this issue, a lifecycle section needs to be added to the end of the main resource declaration, as mentioned in [Terraform docs](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/resources/postgresql_flexible_server):

```plaintext
resource "azurerm_postgresql_flexible_server" "passion_project_db" {
  # previous code

  lifecycle {
    ignore_changes = [
      zone,
    ]
  }
}
```

The remaining two declarations are firewall related:

```plaintext
resource "azurerm_postgresql_flexible_server_firewall_rule" "passion_allow_my_ip_to_pgsql" {
  name             = "passion-allow-my-ip"
  server_id        = azurerm_postgresql_flexible_server.passion_project_db.id
  start_ip_address = data.external.public_ip.result.ip
  end_ip_address   = data.external.public_ip.result.ip
}

resource "azurerm_postgresql_flexible_server_firewall_rule" "azure_services" {
  name             = "AllowAzureServices"
  server_id        = azurerm_postgresql_flexible_server.passion_project_db.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}
```

And just like that, an Azure-backed PostgreSQL instance is provisioned. Don't forget to turn it off after you're done using it, as even with the basic settings used, it incurs some costs.
