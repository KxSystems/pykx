---
title: Registering Custom Operations
description: API for pykx.register
date: October 2024
author: KX Systems, Inc.,
tags: PyKX, register, api
---

# Registering Custom Operations

The purpose of this functionality is to provide an extension mechanism for KDB-X Python allowing users to register extension logic for KDB-X Python.

Specifically this allows users to:

1. Extend the supported conversions from Pythonic types to KDB-X Python objects when using the `#!python pykx.toq` function
2. Extend the supported custom functions on `#!python pykx.Column` objects

::: pykx.register

