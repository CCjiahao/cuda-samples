#ifndef __HELPER_STRING_H__
#define __HELPER_STRING_H__

#include <string.h>
#include <stdlib.h>

inline int stringRemoveDelimiter(char delimiter, const char* string) {
    int string_start = 0;
    while(string[string_start] == delimiter) string_start++;
    if(string_start >= static_cast<int>(strlen(string) - 1)) return 0;
    return string_start;
}

inline bool checkCmdLineFlag(const int argc, const char** argv, const char* string_ref) {
    for (int i = 1; i < argc; i++) {
        int string_start = stringRemoveDelimiter('-', argv[i]);
        const char* string_argv = &argv[i][string_start];
        const char *equal_pos = strchr(string_argv, '=');
        int argv_length = static_cast<int>(equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);
        int length = static_cast<int>(strlen(string_ref));
        if (length == argv_length && !strncasecmp(string_argv, string_ref, length)) {
            return true;
        }
    }
    return false;
}

inline int getCmdLineArgumentInt(const int argc, const char** argv, const char* string_ref) {
    for (int i = 1; i < argc; i++) {
        int string_start = stringRemoveDelimiter('-', argv[i]);
        const char *string_argv = &argv[i][string_start];
        int length = static_cast<int>(strlen(string_ref));
        if (!strncasecmp(string_argv, string_ref, length)) {
            if (length + 1 <= static_cast<int>(strlen(string_argv))) {
                int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                return atoi(&string_argv[length + auto_inc]);
            }
            return 0;
        }
    }
    return 0;
}

inline float getCmdLineArgumentFloat(const int argc, const char** argv, const char* string_ref) {
    for (int i = 1; i < argc; i++) {
        int string_start = stringRemoveDelimiter('-', argv[i]);
        const char *string_argv = &argv[i][string_start];
        int length = static_cast<int>(strlen(string_ref));
        if (!strncasecmp(string_argv, string_ref, length)) {
            if (length + 1 <= static_cast<int>(strlen(string_argv))) {
                int auto_inc = (string_argv[length] == '=') ? 1 : 0;
                return static_cast<float>(atof(&string_argv[length + auto_inc]));
            }
            return 0;
        }
    }
    return 0;
}

inline bool getCmdLineArgumentString(const int argc, const char** argv, const char* string_ref, char** string_retval) {
    for (int i = 1; i < argc; i++) {
        int string_start = stringRemoveDelimiter('-', argv[i]);
        char *string_argv = const_cast<char *>(&argv[i][string_start]);
        int length = static_cast<int>(strlen(string_ref));
        if (!strncasecmp(string_argv, string_ref, length)) {
            *string_retval = &string_argv[length + 1];
            return true;
        }
    }
    *string_retval = NULL;
    return false;
}

#endif // __HELPER_STRING_H__