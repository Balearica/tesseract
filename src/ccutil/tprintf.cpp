/**********************************************************************
 * File:        tprintf.cpp
 * Description: Trace version of printf - portable between UX and NT
 * Author:      Phil Cheatle
 *
 * (C) Copyright 1995, Hewlett-Packard Ltd.
 ** Licensed under the Apache License, Version 2.0 (the "License");
 ** you may not use this file except in compliance with the License.
 ** You may obtain a copy of the License at
 ** http://www.apache.org/licenses/LICENSE-2.0
 ** Unless required by applicable law or agreed to in writing, software
 ** distributed under the License is distributed on an "AS IS" BASIS,
 ** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 ** See the License for the specific language governing permissions and
 ** limitations under the License.
 *
 **********************************************************************/

// Include automatically generated configuration file if running autoconf.
#ifdef HAVE_CONFIG_H
#  include "config_auto.h"
#endif

#include "tprintf.h"

#include "params.h"

#include <climits> // for INT_MAX
#include <cstdarg>
#include <cstdio>

namespace tesseract {

#define MAX_MSG_LEN 2048

INT_VAR(log_level, INT_MAX, "Logging level");

static STRING_VAR(debug_file, "", "File to send tprintf output to");

// Trace printf
void tprintf(const char *format, ...) {
  const char *debug_file_name = debug_file.c_str();
  FILE *debugfp = nullptr; // debug file

  if (debug_file_name == nullptr) {
    // This should not happen.
    return;
  }

  debugfp = fopen(debug_file_name, "a+");
  va_list args;           // variable args
  va_start(args, format); // variable list
  vfprintf(debugfp, format, args);
  // Webassembly build does not always flush properly if not explicitly called for. 
  // See https://emscripten.org/docs/getting_started/FAQ.html#what-does-exiting-the-runtime-mean-why-don-t-atexit-s-run
  va_end(args);
  fclose(debugfp);

}

} // namespace tesseract
