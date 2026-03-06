/* Piece.h
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

#ifndef MINIZERO_SHOGI_PIECE__
#define MINIZERO_SHOGI_PIECE__

//#include "../def.h"
#include <iostream>
#include <cstdint>

namespace minizero::env::shogi {

class Piece {
public:
  static const uint8_t Promotion = 0x08;
  static const uint8_t Empty = 0x20;
  static const uint8_t White = 0x10;

  static const uint8_t HandMask = 0x27;
  static const uint8_t KindMask = 0x2f;

  static const uint8_t Pawn = 0;
  static const uint8_t Lance = 1;
  static const uint8_t Knight = 2;
  static const uint8_t Silver = 3;
  static const uint8_t Gold = 4;
  static const uint8_t Bishop = 5;
  static const uint8_t Rook = 6;
  static const uint8_t King = 7;
  static const uint8_t Tokin = Promotion | Pawn;
  static const uint8_t ProLance = Promotion | Lance;
  static const uint8_t ProKnight = Promotion | Knight;
  static const uint8_t ProSilver = Promotion | Silver;
  static const uint8_t Horse = Promotion | Bishop;
  static const uint8_t Dragon = Promotion | Rook;

  static const uint8_t BPawn = Pawn;
  static const uint8_t BLance = Lance;
  static const uint8_t BKnight = Knight;
  static const uint8_t BSilver = Silver;
  static const uint8_t BGold = Gold;
  static const uint8_t BBishop = Bishop;
  static const uint8_t BRook = Rook;
  static const uint8_t BKing = King;
  static const uint8_t BTokin = Tokin;
  static const uint8_t BProLance = ProLance;
  static const uint8_t BProKnight = ProKnight;
  static const uint8_t BProSilver = ProSilver;
  static const uint8_t BHorse = Horse;
  static const uint8_t BDragon = Dragon;

  static const uint8_t WPawn = White | Pawn;
  static const uint8_t WLance = White | Lance;
  static const uint8_t WKnight = White | Knight;
  static const uint8_t WSilver = White | Silver;
  static const uint8_t WGold = White | Gold;
  static const uint8_t WBishop = White | Bishop;
  static const uint8_t WRook = White | Rook;
  static const uint8_t WKing = White | King;
  static const uint8_t WTokin = White | Tokin;
  static const uint8_t WProLance = White | ProLance;
  static const uint8_t WProKnight = White | ProKnight;
  static const uint8_t WProSilver = White | ProSilver;
  static const uint8_t WHorse = White | Horse;
  static const uint8_t WDragon = White | Dragon;

  static const uint8_t Num = WDragon + 1;
  static const uint8_t Begin = BPawn;
  static const uint8_t End = WDragon + 1;

  static const uint8_t HandNum = Rook + 1;
  static const uint8_t HandBegin = Pawn;
  static const uint8_t HandEnd = Rook + 1;

  static const uint8_t KindNum = Dragon + 1;
  static const uint8_t KindBegin = Pawn;
  static const uint8_t KindEnd = Dragon + 1;

private:

  uint8_t index_;

public:

  constexpr Piece() : index_(Empty) {
  }

  constexpr Piece(uint8_t index) : index_(index) {
  }

  explicit constexpr operator uint8_t() const {
    return index_;
  }
  constexpr uint8_t index() const {
    return index_;
  }

  bool operator==(const Piece& piece) const {
    return index_ == piece.index_;
  }
  bool operator!=(const Piece& piece) const {
    return index_ != piece.index_;
  }

  constexpr bool exists() const {
    return index_ != Empty;
  }
  constexpr bool isEmpty() const {
    return index_ == Empty;
  }

  constexpr Piece hand() const {
    return Piece(index_ & HandMask);
  }
  constexpr Piece promote() const {
    return Piece(index_ | Promotion);
  }
  constexpr Piece unpromote() const {
    return Piece(index_ & ~Promotion);
  }
  constexpr Piece kindOnly() const {
    return Piece(index_ & KindMask);
  }
  constexpr Piece black() const {
    return Piece(index_ & ~White);
  }
  constexpr Piece white() const {
    return Piece(index_ | White);
  }

  constexpr bool isUnpromoted() const {
    return !isPromoted();
  }
  constexpr bool isPromoted() const {
    return index_ & Promotion;
  }
  constexpr bool isBlack() const {
    return !(index_ & (Empty | White));
  }
  constexpr bool isWhite() const {
    return index_ & White;
  }

  constexpr Piece next() const {
    return ((index_ == (Promotion | BGold) - 1U) ||
            (index_ == (Promotion | WGold) - 1U) ||
            (index_ == (Promotion | BKing) - 1U)) ?
           index_ + 2U : index_ + 1U;
  }
  constexpr Piece nextUnsafe() const {
    return Piece(index_ + 1U);
  }

  const char* toString() const {
    static const char* names[] = {
      "fu", "ky", "ke", "gi", "ki", "ka", "hi", "ou",
      "to", "ny", "nk", "ng", "  ", "um", "ry", "  ",
      "Fu", "Ky", "Ke", "Gi", "Ki", "Ka", "Hi", "Ou",
      "To", "Ny", "Nk", "Ng", "  ", "Um", "Ry", "  ",
      "  "
    };
    return names[index_];
  }
  const char* toStringCsa(bool kind_only = false) const {
    static const char* namesCsa[] = {
      "+FU", "+KY", "+KE", "+GI", "+KI", "+KA", "+HI", "+OU",
      "+TO", "+NY", "+NK", "+NG", "   ", "+UM", "+RY", "   ",
      "-FU", "-KY", "-KE", "-GI", "-KI", "-KA", "-HI", "-OU",
      "-TO", "-NY", "-NK", "-NG", "   ", "-UM", "-RY", "   ",
      "   "
    };
    static const char* namesCsaKindOnly[] = {
      "FU", "KY", "KE", "GI", "KI", "KA", "HI", "OU",
      "TO", "NY", "NK", "NG", "  ", "UM", "RY", "  ",
      "FU", "KY", "KE", "GI", "KI", "KA", "HI", "OU",
      "TO", "NY", "NK", "NG", "  ", "UM", "RY", "  ",
      "  "
    };
    return kind_only ? namesCsaKindOnly[kindOnly().index_] : namesCsa[index_];
  }
  static Piece parse(const char* str);
  static Piece parseCsa(const char* str);

};

} // namespace minizero::env::shogi

#define PIECE_EACH(piece)        for (minizero::env::shogi::Piece (piece) = minizero::env::shogi::Piece::Begin; (piece) != minizero::env::shogi::Piece::End; (piece) = (piece).next())
#define PIECE_EACH_UNSAFE(piece) for (minizero::env::shogi::Piece (piece) = minizero::env::shogi::Piece::Begin; (piece) != minizero::env::shogi::Piece::End; (piece) = (piece).nextUnsafe())
#define PIECE_KIND_EACH(piece)   for (minizero::env::shogi::Piece (piece) = minizero::env::shogi::Piece::KindBegin; (piece) != minizero::env::shogi::Piece::KindEnd; (piece) = (piece).nextUnsafe())
#define HAND_EACH(piece)         for (minizero::env::shogi::Piece (piece) = minizero::env::shogi::Piece::HandBegin; (piece) != minizero::env::shogi::Piece::HandEnd; (piece) = (piece).nextUnsafe())

inline bool operator==(uint8_t index, const minizero::env::shogi::Piece& piece) {
  return index == piece.index();
}

inline bool operator!=(uint8_t index, const minizero::env::shogi::Piece& piece) {
  return index != piece.index();
}

inline std::ostream& operator<<(std::ostream& os, const minizero::env::shogi::Piece& piece) {
  os << piece.index();
  return os;
}

#endif //MINIZERO_SHOGI_PIECE__