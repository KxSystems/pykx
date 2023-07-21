system"d .write";

// @kind readme
// @name .write/README.md
// # pykx.q.module.write
// @end

// @kind function
// @fileoverview Splays and writes a q table to disk
// @param dirPath {#hsym|symbol} The path to the root directory into which the splayed table will be written
// @param name {symbol} The name of the table. A directory with this name within the root directory will be created, and will contain the serialized columns of the table
// @param table {table} The table to be splayed and written to disk
// @returns {#hsym} The path to the directory within dirPath which contains the splayed table
.write.splayed:{[dirPath; name; table]
    :(hsym `$"/" sv string dirPath,name) set .Q.en[hsym dirPath;] table
    };

// @kind function
// @fileoverview Writes a q object to disk
// @param path {#hsym|symbol} The path to the file which will store the given data. If a file with this path already exists, it is overwritten.
// @param data {#any} The data to be serialized and written to disk
// @returns {#hsym} The path written to
.write.serialized:{[path; data]
    :hsym[path] set data
    };

// @kind function
// @fileoverview Writes a CSV file given a table
// @param path {#hsym|symbol} The path to the csv file. If a file with this path already exists, it is overwritten.
// @param delimiter {char|null} The single-character delimiter used in the csv file. If null, comma will be used.
// @param table {table} The table to be written as a csv file
// @returns {#hsym} The path written to
.write.csv:{[path; delimiter; table]
    :hsym[path] 0: $[(::)~delimiter;delimiter:",";delimiter] 0: table
    };

// @kind function
// @fileoverview Writes a JSON representation of the given q object
// @param path {#hsym|symbol} The path to the JSON file. If a file with this path already exists, it is overwritten.
// @param data {#any} The q object to be written as a JSON file
// @returns {#hsym} The path written to
.write.json:{[path; data]
    :hsym[path] 0: enlist .j.j data
    };
