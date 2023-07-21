# Modes of Operation

PyKX exists to supersede all previous interfaces between q and Python, as as such it has a few distinct modes of operation. These include:

- Licensed mode
- Unlicensed mode
- Running under q

## Licensed Mode

Licensed mode is the standard mode of operation of PyKX, wherein it is running under a Python process [with a valid q license](../../getting-started/installing.md#licensing-code-execution-for-pykx). All PyKX features are available in this mode.

To provide arguments to q in this mode, the `QARGS` environment variable must be set to a string of command-line arguments. Refer to [the q command-line argument documentation](https://code.kx.com/q/basics/cmdline/) for information about what arguments can be provided.

In addition to the regular arguments taken by q, PyKX accepts its own startup arguments through this mechanism. The following PyKX-specific arguments can be provided:

| Argument       | Description                                                                                                                                       |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `--unlicensed` | Starts PyKX in unlicensed mode. No license check will be performed, and no warning will be emitted at startup if embedded q initialization fails. |
| `--licensed`   | Raise a `PyKXException` (as opposed to emitting a `PyKXWarning`) if embedded q initialization fails.                                              |

For example, if `QARGS` was set to `--licensed -o 1`, then it would ensure that PyKX starts in licensed mode, and embedded q would be provided the [`-o 1`](https://code.kx.com/q/basics/cmdline/#-o-utc-offset), which would set the UTC offset to `UTC+01:00`.

## Unlicensed Mode

Unlicensed mode is a feature-limited mode of operation for PyKX, which has the benefit of not requiring a valid q license (except for the q license required to run the remote q process that PyKX will connect to in this mode).

If the `--unlicensed` flag is provided via the `QARGS` environment variable (as detailed above) then PyKX will start in unlicensed mode regardless of if a valid license is present.

This mode cannot run q embedded within it, and so it lacks the ability to run q code within the local Python process, and also every feature that depends on running q code. Despite this limitation, it provides the following features (which are all also available in licensed mode):

- Conversions from Python to q
- Conversions from q to Python
- [A q IPC interface](../../api/ipc.md)

The IPC interface is key to unlicensed mode, as there is little reason to convert between Python and q unless one can run q code in some way. A [`pykx.QConnection`][pykx.QConnection] instance has largely the same interface and capabilities as the `pykx.q` object, but differs in that it runs all of its q code in a q process over IPC. Through this connection object one can still access a [q console](../../api/console.md), [query interface](../../api/query.md), [context interface](../../api/ctx.md), and of course it can be called to execute q code (in the q process over IPC).

Conversions from Python to q work the same as when running in licensed mode, with the exception of callable Python objects (e.g. functions), which cannot be converted to q in unlicensed mode.

Conversions from q to Python still provide the same `K` objects as in licensed mode, but with the following limitations:

- The `repr` of these objects no longer shows what the object looks like in q, but rather its address in memory, e.g. `pykx.Table._from_addr(0x7f5b72ef8860)`.
- The `str` of these objects is no longer the string obtained by calling [`.Q.s`](https://code.kx.com/q/ref/dotq/#qs-plain-text) on the object, but rather is the same as the `repr`.
- Indexing into `pykx.Collection` objects (i.e. non-atomic q objects) is not supported in unlicensed mode. A `pykx.LicenseException` is raised if this is attempted. All indexing in unlicensed mode should either be
performed within the q server over IPC, or locally into a Python/Numpy/Pandas/PyArrow representation of the
object, rather than into the `pykx.K` instance directly.
- An optimization for `pykx.List.np` (i.e. the Numpy conversion method for q lists) is not applied in unlicensed mode.
- Keyed tables cannot be converted to Numpy in unlicensed mode. A `pykx.LicenseException` is raised if this is attempted.
- The `is_null`, `is_inf`, `has_nulls`, and `has_infs` methods of `K` objects are not supported in unlicensed mode. A `pykx.LicenseException` is raised if they are called.
- Some types cannot be disambiguated, e.g. generic null versus projection null, splayed table versus regular table, etc. - this should not matter in almost every case, but is listed here in the interest of being comprehensive. When these odd types are encountered, they will be exposed as the next highest type in [the `K` type hierarchy](../../api/wrappers.md) that they could be identified as.
- Direct conversions between `pykx.K` types is not possible in unlicensed mode.

Arguments cannot be provided to q via PyKX in this mode, as q is not running within PyKX in this mode. Arguments for q must instead be provided at the command-line when starting the q process that will be connected to.

## Running Under q

Fully described [here](running_under_q.md) the ability to use PyKX within a q session directly is intended to provide the ability to replace embedPy functionally with an updated and more flexible interface. Additionally it provides the ability to use Python functionality within a q environment which does not have the central limitations that exist for PyKX as outlined [here](limitations.md), namely Python code can be used in conjunction with timers and subscriptions within a q/kdb+ ecosystem upon which are reliant on these features of the language.

Similar to the use of PyKX in it's licensed modality PyKX running under q requires a user to have access to an appropriate license containing the `insights.lib.pykx` and `insights.lib.embedq` licensing flags.
