---
layout: post
title: "Building an Azure serverless app"
date: 2024-07-19 15:00:00 -0000
categories: 
    - cloud
    - serverless
    - c#
tags: ["cloud", "azure", "functions", "azure-function", "c#", "open-api"]
---

# Building an Azure serverless app

Going serverless has one main advantage: all the hosting configuration, such as configuring AKS clusters, is eliminated. After years of working on the backend, I decided to give it a try, and... configuring my Azure Function to work as I wanted was harder than anticipated. Maybe it was just my bad luck, but I got lost in the documentation or used advice from outdated blog posts. It took me a couple of hours to get everything working, so I created this blog post to help me initialize similar projects faster in the future.

## Requirements

I wanted this project to look and behave like a standard ASP.NET Core-backed REST API.

## The code

For me, the hardest part of any project is configuration. For this one, I struggled the most with setting up Swagger docs. It seems that you're supposed to do it slightly differently depending on what function runtime you use. I used the new standard (isolated worker), and this set of packages made it all work:

```xml
<ItemGroup>
    <FrameworkReference Include="Microsoft.AspNetCore.App"/>
    <PackageReference Include="Autofac.Extensions.DependencyInjection" Version="9.0.0" />
    <PackageReference Include="Microsoft.Azure.Functions.Worker" Version="1.21.0"/>
    <PackageReference Include="Microsoft.Azure.Functions.Worker.Sdk" Version="1.17.0"/>
    <PackageReference Include="Microsoft.ApplicationInsights.WorkerService" Version="2.22.0"/>
    <PackageReference Include="Microsoft.Azure.Functions.Worker.ApplicationInsights" Version="1.2.0"/>
    <PackageReference Include="Microsoft.Azure.Functions.Worker.Extensions.Http" Version="3.1.0" />
    <PackageReference Include="Microsoft.Azure.Functions.Worker.Extensions.OpenApi" Version="1.5.1" />
    <PackageReference Include="Microsoft.Azure.WebJobs.Extensions.OpenApi.Core" Version="1.5.1" />
</ItemGroup>
```

Here's the `Program.cs` part - I'll show the full definition because it's quite small.

```csharp
using Autofac;
using Autofac.Extensions.DependencyInjection;
using GoingServerless;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Extensions.OpenApi.Extensions;
using Microsoft.Azure.WebJobs.Extensions.OpenApi.Core.Abstractions;
using Microsoft.Azure.WebJobs.Extensions.OpenApi.Core.Configurations;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.OpenApi.Models;

var host = new HostBuilder()
    .UseServiceProviderFactory(new AutofacServiceProviderFactory())
    .ConfigureContainer<ContainerBuilder>(builder =>
    {
        builder.RegisterType<IsTheNumberEven>().AsImplementedInterfaces().SingleInstance();
    })
    .ConfigureServices(services =>
    {
        services.AddApplicationInsightsTelemetryWorkerService();
        services.ConfigureFunctionsApplicationInsights();
        services.AddSingleton<IOpenApiConfigurationOptions>(_ =>
        {
            var options = new OpenApiConfigurationOptions
            {
                Info = new OpenApiInfo
                {
                    Version = DefaultOpenApiConfigurationOptions.GetOpenApiDocVersion(),
                    Title = "Going serverless OpenAPI docs"
                },
                Servers = DefaultOpenApiConfigurationOptions.GetHostNames(),
                OpenApiVersion = DefaultOpenApiConfigurationOptions.GetOpenApiVersion(),
                IncludeRequestingHostName = DefaultOpenApiConfigurationOptions.IsFunctionsRuntimeEnvironmentDevelopment(),
                ForceHttps = DefaultOpenApiConfigurationOptions.IsHttpsForced(),
                ForceHttp = DefaultOpenApiConfigurationOptions.IsHttpForced(),
            };

            return options;
        });
    })
    .ConfigureFunctionsWorkerDefaults(worker =>
    {
        worker.UseMiddleware<ErrorExposingMiddleware>();
    })
    .ConfigureOpenApi()
    .Build();

host.Run();
```

What I like about C# is that most of its libraries are so cleanly written that explaining how a particular method works actually adds noise to the signal. So, in this case, I won't explain each method apart from saying that the OpenAPI configuration included above is what made it all work. There's also the line registering a middleware, so let's get to that next.

```csharp
public class ErrorExposingMiddleware(ILogger<ErrorExposingMiddleware> logger) : IFunctionsWorkerMiddleware
{
    public async Task Invoke(FunctionContext context, FunctionExecutionDelegate next)
    {
        if (!this.IsEnabled())
            await next(context);
        else
        {
            logger.LogInformation($"{nameof(ErrorExposingMiddleware)} is enabled.");
            
            await HandleErrors(context, next);
        }
    }

    private static async Task HandleErrors(FunctionContext context, FunctionExecutionDelegate next)
    {
        try
        {
            await next(context);
        }
        catch (Exception exc)
        {
            var errorMessage = exc.Message;
            var errorBody = DefaultJsonSerializer.Serialize(new
            {
                Error = errorMessage
            });
            var req = await context.GetHttpRequestDataAsync();
            var res = req!.CreateResponse(HttpStatusCode.InternalServerError);

            await res.WriteStringAsync(errorBody);

            context.GetInvocationResult().Value = res;
        }
    }
    
    private bool IsEnabled()
    {
        var middlewareType = GetType();
        var varName = $"{middlewareType.Name}Enabled";
        var enabled = Environment.GetEnvironmentVariable(varName) == "true";

        return enabled;
    }
}
```

Sometimes it's convenient to show backend errors to the client. It's easier to see them in the debugger than to look for them in the backend app logs, and that's what this middleware is for. If the `"ErrorExposingMiddlewareEnabled": "true"` entry is included in the `local.settings.json` `Values` section, this middleware will trigger upon encountering an exception.

Obviously, you can register multiple middlewares, not just one. Since this is all built on top of the AspNetCore framework, there are many similarities. Now, on to the function.

```csharp
public record struct RequestBody(string Name);

public class RandomEndpointFunction(IIsTheNumberEven isTheNumberEven)
{
    [Function(nameof(RandomEndpointFunction))]
    [OpenApiOperation(operationId: "random endpoint", tags: ["RandomEndpoint"],
        Visibility = OpenApiVisibilityType.Important, Description = "Endpoint that will randomly result in an error.")]
    [OpenApiSecurity("function_key", SecuritySchemeType.ApiKey, Name = "code", In = OpenApiSecurityLocationType.Header)]
    [OpenApiRequestBody(MediaTypeNames.Application.Json, typeof(RequestBody))]
    [OpenApiResponseWithBody(statusCode: HttpStatusCode.OK, contentType: MediaTypeNames.Application.Json, bodyType: typeof(RequestBody))]
    [OpenApiResponseWithoutBody(statusCode: HttpStatusCode.BadRequest)]
    public async Task<HttpResponseData> Run(
        [HttpTrigger(AuthorizationLevel.Function, "post", Route = "random-endpoint")] HttpRequestData req,
        FunctionContext context,
        [FromBody] RequestBody body)
    {
        if (!isTheNumberEven.IsIt())
            return req.CreateResponse(HttpStatusCode.BadRequest);

        var response = req.CreateResponse(HttpStatusCode.OK);

        response.Headers.Add("Content-Type", MediaTypeNames.Application.Json);
        await response.WriteStringAsync(DefaultJsonSerializer.Serialize(body));
        
        return response;
    }
}
```

There are a lot of attributes here, most of them used by the Swagger docs generator. One important attribute is the one that precedes the `req` parameter -  `HttpTrigger`. As the name suggests, it declares that the function is triggered by a standard HTTP request. There are many other trigger types, such as `ServiceBusTrigger` or `BlobTrigger`.

## Summary

I couldn't escape the thought that, despite the initial configuration-related issues, this whole app was much easier to create than a similar one targeting AWS Lambda. The tooling was also better. With Microsoft technologies, even when using a non-Microsoft IDE like IntelliJ Rider, only a keypress separates you from reading the code and running it. In contrast, with AWS Lambdas, the SAM CLI and the PyCharm plugin that barely works make it a challenge.
