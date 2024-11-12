---
title: PyKX Real-Time Data Capture
description: Introduction to the PyKX Real-Time Data Capture functionality 
date: July 2024
author: KX Systems, Inc.,
tags: PyKX, q, streaming, publishing, real-time data
---


# Real-Time Data Capture

_This page is an introduction to the PyKX Real-Time Data Capture functionality._ 

The capture, persistence and presentation of high velocity real-time data provides significant challenges to users at all levels from new users attempting to capture this form of data for the first time, to seasoned data-engineers building complex ingestion workflows.

!!! Note "Install q"

        The Real-Time Data Capture functionality provided by PyKX requires you to have access to a `q` executable. A workflow is provided by PyKX to install q as outlined [here](../../../getting-started/installing.md). Alternatively, you will be prompted to install q if not detected when initializing the Real-Time Capture functionality.

The PyKX Real-Time Data Capture functionality described in this documentation provides a framework for users at all levels of their journey to build highly performant real-time systems which can quickly provide users with the following:

- Capture and logging of raw ingested data to facilitate data replay in failure events.
- A Real-Time Database maintaining reference to data from the current day and persisting this data at end of day.
- A Historical Database containing data for all days prior to the current day.

Once you're happy with how your data is being captured and persisted, you can build complex workflows and access more advanced features, such as:

- Add real-time streaming analytics to collect valuable insights from your data and alert on issues in mission critical use-cases.
- Generate real-time and historical query analytics which allow you to derive insights into vast quantities of historical data.
- Control how users query your captured data through centralized gateways which keep users away from mission critical data-ingest.

Below we're breaking down the documentation sections to guide you through the process of generating these systems using PyKX and what to consider while building up your infrastructure.

## Sections

|*#*| **Title**                              | **Description**  |
|---|----------------------------------------|------------------|
|1. |[Start basic ingest](basic.md)          | Build your first data ingestion infrastructure covering the logging of incoming messages, creation of a real-time database and loading of a historical database. |
|2. |[Publish data](publish.md)              | Learn how to publish data to your real-time capture system using Python, q and C. |
|3. |[Subscribe to data](subscribe.md)       | Now that data is flowing to your system, how do you subscribe to new updates? |
|4. |[Real-Time analytics](rta.md)           | Analysing real-time data allows for insights to be derived as you need them. Generate insights into your real-time data and account for common problems. |
|5. |[Custom query APIs](custom_apis.md)     | Querying historical and real-time data using custom Python APIs allows you and the consumers of your data to gain complex insights into your data. |
|6. |[Query access gateways](gateways.md)    | Not all users will have free-form access to query your data. They will instead query via authenticated gateway processes. We will outline why this is useful and how to configure this.|
|7. |[Complex streaming control](complex.md) |Learn how to further edit/manage your streaming workflows with PyKX. Methods include: fine-grained ingest control, process logs and stopping processed. |

!!! Warning "Disclaimer"

        The Real-Time Data Collection functionality provides the necessary tools for users to build complex streaming infrastructures. The generation and management of such workflows rest solely with the users. KX supports only individual elements used to create these workflows, not the end-to-end applications.

## Next steps

- [Start](basic.md) your basic ingest infrastructure.
