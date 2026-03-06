/* Move.cpp
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

#include "move.h"
#include "board.h"
#include <sstream>

namespace minizero::env::shogi {

uint16_t Move::serialize16(const Move& obj) {
  if (obj.isEmpty()) {
    return S16_EMPTY;
  } else if (obj.isHand()) {
    assert(!(obj.piece().isPromoted()));
    uint32_t masked = obj.move_ & (TO | PIECE);
    uint32_t shifted = (masked >> S16_HAND_SHIFT);
    assert(shifted < 0x0800);
    uint16_t data = (uint16_t)(shifted | S16_HAND);
    assert(((uint32_t)data & ~S16_HAND) < 0x0800);
    return data;
  } else {
    uint32_t masked = obj.move_ & (FROM | TO | PROMOTE);
    assert(masked < S16_HAND);
    uint16_t data = (uint16_t)masked;
    assert(!(data & S16_HAND));
    return data;
  }
}

Move Move::deserialize16(uint16_t value, const Board& board) {
  if (value == S16_EMPTY) {
    return empty();
  } else if (value & S16_HAND) {
    Move obj;
    uint32_t masked = (uint32_t)value & ~S16_HAND;
    assert(masked < 0x0800);
    obj.move_ = masked << S16_HAND_SHIFT;
    assert(!obj.piece().isPromoted());
    return obj;
  } else {
    Move obj;
    obj.move_ = value;
    obj.setPiece(board.getBoardPiece(obj.from()));
    return obj;
  }
}

std::string Move::toString() const {
  std::ostringstream oss;

  oss << to().toString();
  oss << piece().toString();
  if (promote()) {
    oss << '+';
  }
  oss << '(';
  if (isHand()) {
    oss << "00";
  } else {
    oss << from().toString();
  }
  oss << ')';

  return oss.str();
}

std::string Move::toStringCsa(bool black) const {
  std::ostringstream oss;

  oss << (black ? '+' : '-');

  if (isHand()) {
    oss << "00";
  } else {
    oss << from().toString();
  }

  oss << to().toString();

  if (promote()) {
    oss << piece().promote().toStringCsa(true);
  } else {
    oss << piece().toStringCsa(true);
  }

  return oss.str();
}

Move Move::parseCsa(const Board& board, const char* s) {
    if (s == nullptr || std::strlen(s) < 6) {
        return Move::empty();
    }
    const char* p = (s[0] == '+' || s[0] == '-') ? s + 1 : s;
    if (!isdigit(p[0]) || !isdigit(p[1]) || !isdigit(p[2]) || !isdigit(p[3])) {
        return Move::empty();
    }
    
    // CSA座標 (file, rank) を文字から数値 (1-9) に変換
    // '2' -> 2, '7' -> 7
    int csaFromFile = p[0] - '0'; 
    int csaFromRank = p[1] - '0';
    int csaToFile = p[2] - '0';
    int csaToRank = p[3] - '0';
    
    // Sunfish の Square(file, rank) コンストラクタを呼び出す
    Square to(csaToFile, csaToRank);
    
    Piece piece_kind = Piece::parseCsa(p + 4); 
    if (piece_kind.isEmpty()) {
        return Move::empty();
    }

    // 駒打ち (00) の場合
    if (p[0] == '0' && p[1] == '0') {
        Piece piece_to_drop = board.isBlack() ? piece_kind.black() : piece_kind.white();
        return Move(piece_to_drop, to); 
    }
    
    // 盤上の移動
    Square from(csaFromFile, csaFromRank);
    Piece piece_on_board = board.getBoardPiece(from); 
    
    if (piece_on_board.isEmpty()) { 
        return Move::empty(); 
    }
    if (piece_on_board.isBlack() != board.isBlack()) { 
        return Move::empty(); 
    }

    bool promote = (piece_on_board.kindOnly() != piece_kind.kindOnly());
    
    return Move(piece_on_board, from, to, promote);
}

} // namespace minizero::env::shogi