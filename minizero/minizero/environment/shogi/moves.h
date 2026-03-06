/* moves.h
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

#ifndef MINIZERO_SHOGI_MOVES__
#define MINIZERO_SHOGI_MOVES__

#include "board.h"
#include <algorithm>

namespace minizero::env::shogi {

template <int capacity>
class TempMoves {
private:

  Move moves_[capacity];
  int size_;

public:

  using iterator = Move*;
  using const_iterator = const Move*;

  TempMoves() : size_(0) {
  }

  void clear() { size_ = 0; }
  int size() const { return size_; }

  void add(const Move& move) {
    moves_[size_++] = move;
  }

  void remove(int index) {
    moves_[index] = moves_[--size_];
  }
  iterator remove(iterator ite) {
    (*ite) = moves_[--size_];
    return ite;
  }
  void removeStable(int index) {
    removeStable(begin()+index);
  }
  iterator removeStable(iterator ite) {
    for (auto itmp = ite+1; itmp != end(); itmp++) {
      *(itmp-1) = *(itmp);
    }
    size_--;
    return ite;
  }
  void removeAfter(int index) {
    size_ = index;
  }
  void removeAfter(iterator ite) {
    size_ = (int)(ite - moves_);
  }

  // random accessor
  Move& get(int index) { return moves_[index]; }
  const Move& get(int index) const { return moves_[index]; }
  Move& operator[](int index) { return moves_[index]; }
  const Move& operator[](int index) const { return moves_[index]; }

  // iterator
  iterator begin() { return moves_; }
  const_iterator begin() const { return moves_; }
  iterator end() { return moves_ + size_; }
  const_iterator end() const { return moves_ + size_; }

  iterator find(const Move& move) {
    for (auto ite = begin(); ite != end(); ite++) {
      if (ite->equals(move)) {
        return ite;
      }
    }
    return end();
  }

};

using Moves = TempMoves<1024>;

} // namespace minizero::env::shogi

#endif //MINIZERO_SHOGI_MOVES__