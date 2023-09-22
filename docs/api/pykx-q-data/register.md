# Registering Custom Conversions

The purpose of this functionality is to provide an extension mechanism for PyKX allowing users to register extension logic for handling conversions from Pythonic types to create PyKX objects when using the `pykx.toq` function or any internal functionality which makes use of this conversion mechanism.

::: pykx.register
