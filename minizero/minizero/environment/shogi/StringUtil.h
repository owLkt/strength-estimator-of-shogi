/* StringUtil.h
 *
 * Original Author: Kubo Ryosuke
 * Copyright (c) 2015 Ryosuke Kubo
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef SUNFISH_STRINGUTIL__
#define SUNFISH_STRINGUTIL__

#include <sstream>
#include <iomanip>
#include <cstdint>

namespace minizero::env::shogi {

class StringUtil {
private:

  StringUtil() {}

public:

  static std::string stringify(unsigned u32) {
    std::ostringstream oss;
    oss << std::setw(8) << std::setfill('0') << std::hex << u32;
    return oss.str();
  }

  static std::string stringify(uint64_t u64) {
    return stringify((unsigned)(u64>>32)) + stringify((unsigned)u64);
  }

  static std::string chomp(const std::string& line) {
    for (int index = (int)line.length()-1; index >= 0; index--) {
      if (line.at(index) != '\n') {
        return line.substr(0, index + 1);
      }
    }
    return line;
  }

};

} // namespace sunfish

#endif //SUNFISH_STRINGUTIL__