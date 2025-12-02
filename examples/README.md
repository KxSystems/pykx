# Examples

This folder contains a number of subfolders/files containing example uses of the PyKX interface. These examples are not intended to show fully all that can be achieved by the interface but rather act as inspiration for some of the potential use cases of the interface.

This `README.md` is intended to act as a guide for the contents within each of the individual examples explaining the use case and how the example should be run

**Note:**

The `archive` folder contains a number of currently unsupported examples which had previously been demonstrable within the interface. As such until such time as these have been updated in line with the new interface these are not intended to be run.

## quickdemo.txt

This file provides a sample 'script' outlining a brief demonstration of the functionality that is provided by the interface. This can be used for reference when providing a demo of the interface's capabilities from terminal.

## notebooks/

The notebooks folder contains any Jupyter notebook examples which may be useful for users.

### Interface_outline.ipynb

This outlines the use of a significant portion of the interface functionality.

This example is the best centralised representation of the wide ranging functionality that is available with the interface and should be the first stop for new users to the interface. This notebook contains the following sections

1. Initializing the library
2. Generating q objects
3. Converting q to Python
4. Interacting with K objects
5. Extension modules
6. Querying Interface
7. IPC communication

## ipc/

The ipc folder contains any IPC specific examples which may be useful for users.

This IPC example provided at present outlines the use of PyKX as a real-time engine, ingesting data as a subscriber to an external q process, evaluating an analytic written in Python and publishing the result of this analytic to another external q process.

This example works in the presence or absence of a q license and is fully outlined in `ipc/README.md`

## extensions/

The extensions folder contains a file `extensions.q` which should be placed in `$QHOME` when running the demonstrations outlined in `quickdemo.txt` and `notebooks/Interface_outline.q`.

## compress_and_encrypt/

The compress and encrypt demo outlined by the `compress_and_encrypt.py` file and envoked from the associated folder using by running

```bash
$ python compress_and_encrypt.py
```

This example provides an example of how a user of PyKX can make use of the supported compression algorithms provided by q

* gzip
* snappy
* lz4hc

The example defines a `compress` function which is used to apply the above compressions on a q trade table of length 1 million.

For each invocation the following information is printed to terminal summarizing the completed compression

```
Writing in-memory trades table with gzip: {
    "compressedLength": 12503352,
    "uncompressedLength": 36666552,
    "algorithm": 2,
    "logicalBlockSize": 17,
    "zipLevel": 9
}
```
