/**
 *  Lab6 - TD learning for 2048
 *  此程式透過TD learning 和TD after-state learning來學習
 *  程式部份參考自 NCTU CGI 釋出的範例程式
 * 
 */

#include <iostream>
#include <algorithm>
#include <functional>
#include <iterator>
#include <vector>
#include <array>
#include <limits>
#include <numeric>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>

// 定義output stream
std::ostream& info  = std::cout;
std::ostream& error = std::cerr;
std::ostream& debug = *(new std::ofstream);

/**
 * 64-bit 定義 2048盤面
 * 舉個例子，當index為 0x4312752186532731ull，則盤面如下：
 * +------------------------+
 * |     2     8   128     4|
 * |     8    32    64   256|
 * |     2     4    32   128|
 * |     4     2     8    16|
 * +------------------------+
 */

class board {
    public:
        // Constructor        
        board(uint64_t raw = 0) : raw(raw) {}
        board(const board& b) = default;
        operator uint64_t() const { return raw;}

        /**
         *  定義基本操作 - Get or Set 
         */
        // Get a 16-bit row, i為第幾個row你想要獲取
        int fetch(int i) const {
            return ((raw >> (i << 4)) & 0xffff);
        }
        // Set a 16-bit row
        void place(int i, int r) {
            raw = (raw & ~(0xffffULL << (i << 4))) | (uint64_t(r & 0xffff) << (i << 4));
        }
        // Get a 4-bit tile
        int at(int i) const {
            return (raw >> (i << 2)) & 0x0f;
        }
        // Set a 4-bit tile
        void set(int i, int t) {
            raw = (raw & ~(0x0fULL << (i << 2))) | (uint64_t(t & 0x0f) << (i << 2));
        }

        // Operator overloadding
        board& operator =  (const board& b) = default;
        bool   operator == (const board& b) const { return raw == b.raw;}
        bool   operator <  (const board& b) const { return raw < b.raw;}
        bool   operator != (const board& b) const { return !(*this == b);}
        bool   operator >  (const board& b) const { return b < *this;}
        bool   operator <= (const board& b) const { return !(b < *this);}
        bool   operator >= (const board& b) const { return !(*this < b);}

        // 定義cout輸出格式
        friend std::ostream& operator <<(std::ostream& out, const board& b) {
	    	char buff[32];
	    	out << "+------------------------+" << std::endl;
	    	for (int i = 0; i < 16; i += 4) {
	    		snprintf(buff, sizeof(buff), "|%6u%6u%6u%6u|",
	    			(1 << b.at(i + 0)) & -2u, // use -2u (0xff...fe) to remove the unnecessary 1 for (1 << 0)
	    			(1 << b.at(i + 1)) & -2u,
	    			(1 << b.at(i + 2)) & -2u,
	    			(1 << b.at(i + 3)) & -2u);
	    		out << buff << std::endl;
	    	}
	    	out << "+------------------------+" << std::endl;
	    	return out;
	    }

        /** 
         * 初始化盤面
         * 將盤面清空後，隨機跳出兩個tile
         */
        void init() {
            raw = 0;
            popup();
            popup();
        }

        /**
         * 新增一個隨機的tile，
         * 90%機率產生2 (code為1)，
         * 10%機率產生4 (code為2)，
         * 如果盤面是滿的，則不做任何事
         */
        void popup() {
            int space[16], num = 0;
		    for (int i = 0; i < 16; i++)
		    	if (at(i) == 0) {
		    		space[num++] = i;
		    	}
		    if (num)    // 如果盤面不是滿的
		    	set(space[rand() % num], rand() % 10 ? 1 : 2);
        }

        
        // ------------------------------------------
        // 定義移動相關操作
        // ------------------------------------------
        /**
         * 移動的wrapper，
         * 根據輸入的code來進行盤面移動
         * code介於[0, 3]
         */ 
        int move(int opcode) {
            switch (opcode) {
                case 0: return move_up();
                case 1: return move_right();
                case 2: return move_down();
                case 3: return move_left();
                default: return -1;
            }
        }

        /**
         * 盤面整個往左移動
         * 總共有4個row，對每一個row獲取盤面後，查找並設置往左移動後的結果
         */
        int move_left() {
	    	uint64_t move = 0;
	    	uint64_t prev = raw;
	    	int score = 0;
	    	lookup::find(fetch(0)).move_left(move, score, 0);
	    	lookup::find(fetch(1)).move_left(move, score, 1);
	    	lookup::find(fetch(2)).move_left(move, score, 2);
	    	lookup::find(fetch(3)).move_left(move, score, 3);
	    	raw = move;
	    	return (move != prev) ? score : -1;
	    }

        /**
         * 盤面整個往右移動
         * 總共有4個row，對每一個row獲取盤面後，查找並設置往右移動後的結果
         */
        int move_right() {
	    	uint64_t move = 0;
	    	uint64_t prev = raw;
	    	int score = 0;
	    	lookup::find(fetch(0)).move_right(move, score, 0);
	    	lookup::find(fetch(1)).move_right(move, score, 1);
	    	lookup::find(fetch(2)).move_right(move, score, 2);
	    	lookup::find(fetch(3)).move_right(move, score, 3);
	    	raw = move;
	    	return (move != prev) ? score : -1;
	    }

        /**
         * 盤面整個往上移動
         * 先針對整個盤面做順時針旋轉後，在往左移動，再轉回來
         */
        int move_up() {
	    	rotate_right();
	    	int score = move_right();
	    	rotate_left();
	    	return score;
	    }

        /**
         * 盤面整個往下移動
         * 先針對整個盤面做順時針旋轉後，在往右移動，再轉回來
         */
	    int move_down() {
	    	rotate_right();
	    	int score = move_left();
	    	rotate_left();
	    	return score;
	    }

        /**
         * 定義順時針旋轉、逆時針旋轉和reverse操作
         */
        void rotate_right() { transpose(); mirror(); }  // clockwise
	    void rotate_left() { transpose(); flip(); }     // counterclockwise
	    void reverse() { mirror(); flip(); }

        /**
         * 對盤面做transpose
         */
        void transpose() {
	    	raw = (raw & 0xf0f00f0ff0f00f0fULL) | ((raw & 0x0000f0f00000f0f0ULL) << 12) | ((raw & 0x0f0f00000f0f0000ULL) >> 12);
	    	raw = (raw & 0xff00ff0000ff00ffULL) | ((raw & 0x00000000ff00ff00ULL) << 24) | ((raw & 0x00ff00ff00000000ULL) >> 24);
	    }

        /**
         * 對盤面做鏡射 (水平翻轉)
         */
        void mirror() {
	    	raw = ((raw & 0x000f000f000f000fULL) << 12) | ((raw & 0x00f000f000f000f0ULL) << 4)
	    	    | ((raw & 0x0f000f000f000f00ULL) >> 4) | ((raw & 0xf000f000f000f000ULL) >> 12);
	    }

        /**
         * 對盤面做垂直翻轉
         */
        void flip() {
	    	raw = ((raw & 0x000000000000ffffULL) << 48) | ((raw & 0x00000000ffff0000ULL) << 16)
	    	    | ((raw & 0x0000ffff00000000ULL) >> 16) | ((raw & 0xffff000000000000ULL) >> 48);
	    }

        /**
         * 對盤面做r次的逆時針旋轉
         */
        void rotate(int r = 1) {
	    	switch (((r % 4) + 4) % 4) {
	    	    default:
	    	    case 0: break;
	    	    case 1: rotate_right(); break;
	    	    case 2: reverse(); break;
	    	    case 3: rotate_left(); break;
	    	}
	    }

    private:
        /**
         * 定義盤面
         */
        uint64_t raw;

        /**
         * 定義 look-up table 以加速移動盤面的運算
         */
        struct lookup {
            int raw;    // 一個16-bit的row
            int left;   // 往左移動
            int right;  // 往右移動
            int score;  // 總得分 (total reward)

            /**
             * 初始化 look-up table
             */
            void init(int r) {
                raw = r;

			    int V[4] = { 
                    (r >> 0) & 0x0f, 
                    (r >> 4) & 0x0f, 
                    (r >> 8) & 0x0f, 
                    (r >> 12) & 0x0f 
                };
			    int L[4] = { V[0], V[1], V[2], V[3] };
			    int R[4] = { V[3], V[2], V[1], V[0] }; // mirrored

			    score = mvleft(L);
			    left = ((L[0] << 0) | (L[1] << 4) | (L[2] << 8) | (L[3] << 12));

			    score = mvleft(R); 
                std::reverse(R, R + 4);
			    right = ((R[0] << 0) | (R[1] << 4) | (R[2] << 8) | (R[3] << 12));
            }

            // Look-up table constructor
            lookup() {
		    	static int row = 0;
		    	init(row++);
		    }

            static int mvleft(int row[]) {
		    	int top = 0;
		    	int tmp = 0;
		    	int score = 0;

		    	for (int i = 0; i < 4; i++) {
		    		int tile = row[i];
		    		if (tile == 0) continue;
		    		row[i] = 0;
		    		if (tmp != 0) {
		    			if (tile == tmp) {
		    				tile = tile + 1;
		    				row[top++] = tile;
		    				score += (1 << tile);
		    				tmp = 0;
		    			} else {
		    				row[top++] = tmp;
		    				tmp = tile;
		    			}
		    		} else {
		    			tmp = tile;
		    		}
		    	}
		    	if (tmp != 0) row[top] = tmp;
		    	return score;
		    }

            /**
             * 定義基本移動操作 - 針對一個row來運算
             */
            void move_left(uint64_t& raw, int& sc, int i) const {
		    	raw |= uint64_t(left) << (i << 4);
		    	sc += score;
		    }

		    void move_right(uint64_t& raw, int& sc, int i) const {
		    	raw |= uint64_t(right) << (i << 4);
		    	sc += score;
		    }

            static const lookup& find(int row) {
		    	static const lookup cache[65536];
		    	return cache[row];
		    }
        };
};

int main() {

}