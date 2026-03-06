/* Hand.h
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

#ifndef MINIZERO_SHOGI_HAND__
#define MINIZERO_SHOGI_HAND__

#include "./Piece.h"
#include <cstdint>
#include <cstring>
#include <cassert>

namespace minizero::env::shogi {

class Hand {
private:

  uint8_t counts_[Piece::HandNum];

public:

  constexpr Hand() : counts_{} {
  }

  void init() {
    memset(counts_, 0, sizeof(counts_));
  }

  int inc(const Piece& piece) {
    return incUnsafe(piece.kindOnly().unpromote());
  }
  int incUnsafe(const Piece& piece) {
    assert(!piece.isPromoted());
    assert(!piece.isWhite());
    assert(counts_[piece.index()] < 18);
    assert(piece.index() == Piece::Pawn || counts_[piece.unpromote().index()] < 4);
    assert(piece.index() <= Piece::Gold || counts_[piece.unpromote().index()] < 2);
    return ++counts_[piece.index()];
  }

  int dec(const Piece& piece) {
    return decUnsafe(piece.kindOnly().unpromote());
  }
  int decUnsafe(const Piece& piece) {
    assert(!piece.isPromoted());
    assert(!piece.isWhite());
    assert(counts_[piece.index()] > 0);
    return --counts_[piece.index()];
  }

  constexpr int get(const Piece& piece) const {
    return counts_[piece.kindOnly().unpromote().index()];
  }
  constexpr int getUnsafe(const Piece& piece) const {
    return counts_[piece.index()];
  }
  void set(const Piece& piece, int count) {
    counts_[piece.kindOnly().unpromote().index()] = (int8_t)count;
  }
};

} // namespace minizero::env::shogi

#endif //MINIZERO_SHOGI_HAND__