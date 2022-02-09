// Copyright 2007 Edd Dawson.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//embd ns=ns1::ns2 source=files.cpp header=files.hpp file1.jpg [@ images/file1.jpg]

#include <algorithm>
#include <cassert>
#include <cctype>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __BORLANDC__
#pragma warn -8091 // template argument InputIterator passed to 'find_if' is a output iterator: input iterator required in function trim(const string &)
#pragma warn -8013 // Possible use of 'iic' before definition in function valid_identifier(const string &)
#endif

using namespace std;

//-------------------------------------------------------------------------------------------------
// Data structures
//-------------------------------------------------------------------------------------------------

struct file_data
{
    file_data(const string &src, const string &tgt) : source(src), destination(tgt) { }
    file_data(const string &src) : source(src), destination(src) { }

    string source;
    string destination;
};

struct order_by_destination : std::binary_function<file_data, file_data, bool>
{
    bool operator() (const file_data &lhs, const file_data &rhs) const
    {
        return lhs.destination < rhs.destination;
    }
};

struct compare_by_destination : std::binary_function<file_data, file_data, bool>
{
    bool operator() (const file_data &lhs, const file_data &rhs) const
    {
        return lhs.destination == rhs.destination;
    }
};

typedef map<string, string> environment;
typedef std::pair<string, string> str_pair;

//-------------------------------------------------------------------------------------------------
// Exception classes
//-------------------------------------------------------------------------------------------------

class command_line_error : public std::runtime_error
{
    public:
        command_line_error(const std::string &err) : std::runtime_error(err) { }
        ~command_line_error() throw() { }
};

class file_open_error : public std::runtime_error
{
    public:
        file_open_error(const std::string &err) : std::runtime_error(err) { }
        ~file_open_error() throw() { }
};

class invalid_identifier_error : public std::runtime_error
{
    public:
        invalid_identifier_error(const std::string &err) : std::runtime_error(err) { }
        ~invalid_identifier_error() throw() { }
};


//-------------------------------------------------------------------------------------------------
// Parsing and string manipulation functions
//-------------------------------------------------------------------------------------------------

template<typename Container, typename Element>
bool contains(const Container &c, const Element &e)
{
    return std::find(c.begin(), c.end(), e) != c.end();
}

struct not_space : std::unary_function<bool, char>
{
    bool operator() (char ch) { return !::isspace(ch); }
};

string trim(const string &s)
{
    typedef string::const_iterator iter;

    iter front = find_if(s.begin(), s.end(), not_space());
    iter back = find_if(s.rbegin(), s.rend(), not_space()).base();

    return (back < front) ? string() : string(front, back);
}

str_pair split(const string &s, char about)
{
    typedef string::const_iterator iter;
    iter b = s.begin(), e = s.end(), pos = find(b, e, about);

    if (pos == e) return str_pair(s, "");

    return str_pair( trim(string(s.begin(), pos)), trim(string(pos + 1, e)) );
}

template<typename InputIter>
void parse_command_line(InputIter begin, InputIter end, vector<file_data> &files, environment &env)
{
    bool expecting_target = false;
    string filename;

    while (begin != end)
    {
        string arg = trim(*begin++);

        if (expecting_target)
        {
            if (contains(arg, '=')) throw command_line_error("file name after '@' is an assignment");
            else if (arg == "@") throw command_line_error("found '@' following '@'");

            files.push_back(file_data(filename, arg));
            filename.clear();
            expecting_target = false;
        }
        else if (contains(arg, '='))
            env.insert(split(arg, '='));
        else if (arg == "@")
        {
            if (filename.empty()) throw command_line_error("found '@' without a preceding file name");
            expecting_target = true;
        }
        else
        {
            if (!filename.empty())
                files.push_back(filename);
            filename = arg;
        }
    }

    if (expecting_target)
        throw command_line_error("found '@' without a target file name");

    if (!filename.empty())
        files.push_back(filename);
}

void set_default(environment &env, const string &key, const string &value, bool allow_empty = false)
{
    environment::iterator p = env.find(key);

    if (p == env.end() || (!allow_empty && p->second.empty()))
        env[key] = value;
}

struct invalid_identifier_char : std::unary_function<int, bool>
{
    bool operator() (int ch) const  { return ch != '_' && !isalnum(ch); }
};

bool valid_identifier(const std::string &identifier)
{
    invalid_identifier_char iic;

    return (!identifier.empty() && find_if(identifier.begin(), identifier.end(), iic) == identifier.end()) &&
           !isdigit(identifier[0]);
}

void split_namespace(const string &ns, vector<string> &parts)
{
    typedef string::size_type sz_t;

    sz_t start = 0, end = 0, n = ns.size();

    do
    {
        end = ns.find("::", start);
        end = min(end, n);
        if (start != end)
        {
            string part = ns.substr(start, end - start);
            if (!valid_identifier(part))
                throw invalid_identifier_error("'" + part + "' is not a valid namespace identifier");
            parts.push_back(part);
        }
        start = end + 2;
    }
    while (end != n);
}

//-------------------------------------------------------------------------------------------------
// Canned text dumping functions
//-------------------------------------------------------------------------------------------------

const char *indentation = "    ";

template<unsigned N>
void write_lines(std::ostream &out, const char * (&lines)[N], const char *indent = "")
{
    const char **p = lines;
    const char **e = lines + N;

    while (p != e)
        out << indent << *p++ << '\n';
}

void write_header_header(std::ostream &out)
{
    const char *lines[] =
    {
#include "segments/header_header.hpp"
    };

    write_lines(out, lines);
}

void write_header_guts(std::ostream &out, bool indent)
{
    const char *lines[] =
    {
#include "segments/header_guts.hpp"
    };

    write_lines(out, lines, indent ? indentation : "");
}

void write_source_header(std::ostream &out)
{
    const char *lines[] =
    {
#include "segments/source_header.hpp"
    };

    write_lines(out, lines);
}

void write_source_pre_data(std::ostream &out)
{
    const char *lines[] =
    {
#include "segments/source_pre_data.hpp"
    };

    write_lines(out, lines);
}

void write_source_pre_index(std::ostream &out)
{
    const char *lines[] =
    {
#include "segments/source_pre_index.hpp"
    };

    write_lines(out, lines);
}

void write_source_post_index(std::ostream &out)
{
    const char *lines[] =
    {
#include "segments/source_post_index.hpp"
    };

    write_lines(out, lines);
}

void write_source_class_definitions(std::ostream &out, bool indent)
{
    const char *lines[] =
    {
#include "segments/source_class_definitions.hpp"
    };

    write_lines(out, lines, indent ? indentation : "");
}

//-------------------------------------------------------------------------------------------------
// Other dumping functions
//-------------------------------------------------------------------------------------------------

void write_namespace_opening(std::ostream &out, const vector<string> &ns_parts)
{
    for (unsigned i = 0, n = ns_parts.size(); i != n; ++i)
        out << "namespace " << ns_parts[i] << "\n{\n";
}

void write_namespace_close(std::ostream &out, const vector<string> &ns_parts)
{
    unsigned i = ns_parts.size();

    while (i--)
        out << "} // close namespace " << ns_parts[i] << "\n";
}

// Returns the size of the file in bytes
size_t write_file_data(std::ostream &out, const std::string &src, const std::string &dst)
{
    std::ifstream in(src.c_str(), std::ios::binary);

    if (!in) throw file_open_error("failed to open " + src);

    size_t bytes = 0;
    char ch = 0;

    out << "        // " << dst << "\n        ";
    while (in.get(ch))
    {
        if (bytes != 0) out << ", ";

        // 20 bytes per row
        if (bytes > 0 && bytes % 20 == 0) out << "\n        ";

        unsigned char uch = *reinterpret_cast<unsigned char *>(&ch);

        // we use hex formatting else msvc complains
        out << "'\\x" << std::hex << unsigned(uch) << '\'';

        ++bytes;
    }

    return bytes;
}

void export_header(environment &env, const vector<string> &ns_parts)
{
    std::string loc = env["header_dir"] + '/' + env["header"];
    ofstream out(loc.c_str(), std::ios::binary);

    time_t now = time(0);
    char timestr[20];
    strftime(timestr, sizeof timestr, "%H%M_%d%m%Y", localtime(&now));

    string guard("EMBEDDED_FILES_");
    guard += timestr;

    out << "#ifndef " << guard << '\n';
    out << "#define " << guard << "\n\n";

    write_header_header(out);
    write_namespace_opening(out, ns_parts);
    write_header_guts(out, !ns_parts.empty());
    write_namespace_close(out, ns_parts);

    out << "\n#endif // " << guard << '\n';
}


void export_source(environment &env, const vector<string> &ns_parts, const vector<file_data> &files)
{
    std::string loc = env["source_dir"] + '/' + env["source"];
    ofstream out(loc.c_str(), std::ios::binary);

    write_source_header(out);
    out << "#include \"" << env["header"] << "\"\n";
    write_source_pre_data(out);

    vector<size_t> file_lengths;

    for (size_t i = 0, n = files.size(); i != n; ++i)
    {
        size_t len = write_file_data(out, files[i].source, files[i].destination);
        file_lengths.push_back(len);

        if (i != n - 1) out << ',';
        out << "\n\n";
    }
    out << dec;

    write_source_pre_index(out);

    assert(file_lengths.size() == files.size());
    size_t offset = 0;

    for (size_t i = 0, n = files.size(); i != n; ++i)
    {
        out <<  "        { data + " << offset <<
                ", data + " << (offset + file_lengths[i]) <<
                ", \"" << files[i].destination << "\" }";

        out << ((i == n - 1) ? "\n" : ",\n");
        offset += file_lengths[i];
    }

    write_source_post_index(out);
    write_namespace_opening(out, ns_parts);
    write_source_class_definitions(out, !ns_parts.empty());
    write_namespace_close(out, ns_parts);
}

int main(int argc, char **argv)
{
    try
    {
        environment env;
        vector<file_data> files;

        parse_command_line(argv + 1, argv + argc, files, env);

        // Check we don't have multiple files with the same destination
        // I guess multiple files with the same source should be allowed?
        typedef vector<file_data>::iterator fditer_t;
        std::sort(files.begin(), files.end(), order_by_destination());
        fditer_t b = files.begin();
        fditer_t e = files.end();

        if ((b = std::unique(b, e, compare_by_destination())) != e)
            throw command_line_error("multiple files specified with destination name: " + b->destination);

        // Set some default values in the environment if they don't already exist
        set_default(env, "header", "embd.hpp");
        set_default(env, "source", "embd.cpp");
        set_default(env, "namespace", "embd", true);
        set_default(env, "source_dir", ".");
        set_default(env, "header_dir", ".");

        // get the parts of the user-specified namespace, delimited by ::
        vector<string> nsparts;
        split_namespace(env["namespace"], nsparts);

        export_header(env, nsparts);
        export_source(env, nsparts, files);
    }
    catch(const std::exception &ex)
    {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
