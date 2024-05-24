.pykx.util.loadfile[;"csvutil.q"]{x sv (-1 _ x vs y)}[$[.z.o~`w64;"\\";"/"]; (value{})6];

system"d .read";

// @kind readme
// @name .read/README.md
// # pykx.q.module.read
// @end

// @kind function
// @fileoverview Loads a CSV file as a table; column types are guessed if not provided
// @param path {#hsym|symbol} The path to the csv file
// @param types {(char)|string|null} A list/string of uppercase type characters representing the types, or null. Space is used to drop the associated column. Null is used to guess the type of the associated column.
// @param delimiter {char|null} The single-character delimiter used in the csv file. If null, comma will be used.
// @param asTable {bool} Whether the first line of the csv file should be interpreted as column names. If true, a table will be returned. Otherwise, a list of vectors of columnar data will be returned.
// @returns {table|#any[][]} The data from the csv file as a table or list of vectors depending on the value of the asTable parameter
.read.csv:{[path; types; delimiter; asTable]
    guessedTypes:.csv.info[path]`t;
    :($[(::)~types;guessedTypes;1_?[1b,(::)~'types;guessedTypes;(::),types]];$[asTable;enlist;::]$[(::)~delimiter;delimiter:",";delimiter]) 0: hsym path
    };

// @kind function
// @fileoverview Loads a file of typed data with fixed-width fields. It is expected that there will either be a newline after every record, or none at all.
// @param path {#hsym|symbol} The path to the fixed-width data file
// @param types {string} A string of uppercase type characters representing the types. Space is used to drop the associated column.
// @param widths {long[]} The widths of the fields
// @returns {#any[]|#any[][]} A vector or list of vectors representating the data
.read.fixed:{[path; types; widths]
    :(types;widths) 0: path
    };

// @kind function
// @fileoverview Loads a json file as a q object. Serialization/deserialization to/from JSON may not preserve q datatype
// @param path {#hsym|symbol} The path to the json file
// @returns {table} The JSON object converted to its closest q analogue
.read.json:{[path]
    :.j.k raze read0 path
    };

// @kind function
// @fileoverview Loads a q table
// @param path {#hsym|symbol} The path to the table file/directory
// @returns {table} The table stored at the given path
.read.qtab:{[path]
    :get hsym path
    };
