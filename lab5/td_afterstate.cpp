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
// std::ostream& debug = std::cout;

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

/**
 * 定義N-tuple network的權重表
 */
class feature {
    public:
        // Constructor and destructor
        feature(size_t len) : length(len), weight(alloc(len)) {}
        feature(feature&& f) : length(f.length), weight(f.weight) { f.weight = nullptr; }
        feature(const feature& f) = delete;
        virtual ~feature() {delete[] weight;}

        // Operator overloadding
        feature& operator =  (const feature& f) = delete;
        float&   operator [] (size_t i) { return weight[i]; }
        float    operator [] (size_t i) const { return weight[i]; }

        // Else
        size_t size() const { return length; }

        // The other virtual function that should be implement in the AI bot
        virtual float estimate(const board& b) const = 0;
        virtual float update(const board& b, float u) = 0;
        virtual std::string name() const = 0;

        /**
	     * dump the detail of weight table of a given board
	     */
	    virtual void dump(const board& b, std::ostream& out = info) const {
	    	out << b << "estimate = " << estimate(b) << std::endl;
	    }

	    friend std::ostream& operator <<(std::ostream& out, const feature& w) {
	    	std::string name = w.name();
	    	int len = name.length();
	    	out.write(reinterpret_cast<char*>(&len), sizeof(int));
	    	out.write(name.c_str(), len);
	    	float* weight = w.weight;
	    	size_t size = w.size();
	    	out.write(reinterpret_cast<char*>(&size), sizeof(size_t));
	    	out.write(reinterpret_cast<char*>(weight), sizeof(float) * size);
	    	return out;
	    }

	    friend std::istream& operator >>(std::istream& in, feature& w) {
	    	std::string name;
	    	int len = 0;
	    	in.read(reinterpret_cast<char*>(&len), sizeof(int));
	    	name.resize(len);
	    	in.read(&name[0], len);
	    	if (name != w.name()) {
	    		error << "unexpected feature: " << name << " (" << w.name() << " is expected)" << std::endl;
	    		std::exit(1);
	    	}
	    	float* weight = w.weight;
	    	size_t size;
	    	in.read(reinterpret_cast<char*>(&size), sizeof(size_t));
	    	if (size != w.size()) {
	    		error << "unexpected feature size " << size << "for " << w.name();
	    		error << " (" << w.size() << " is expected)" << std::endl;
	    		std::exit(1);
	    	}
	    	in.read(reinterpret_cast<char*>(weight), sizeof(float) * size);
	    	if (!in) {
	    		error << "unexpected end of binary" << std::endl;
	    		std::exit(1);
	    	}
	    	return in;
	    }

    protected:
        // 表格長度和權重表指標
        size_t length;
        float* weight;

        /**
         * 分配權重表的記憶體，大小為num
         */
        static float* alloc(size_t num){
            static size_t total = 0;
            static size_t limit = (1 << 30) / sizeof(float);    // 最多分配1G
            try{
                total += num;
                if (total > limit) 
                    throw std::bad_alloc();
                return new float[num];
            } catch (std::bad_alloc&) {
                error << "You cannot allocate the memory more than 1G!" << std::endl;
                std::exit(-1);
            }
            return nullptr;
        }
};

/**
 * 定義pattern （繼承自feature）
 * 輸入的參數代表位置
 * 版面定義如下：
 * 
 * index:
 *  0  1  2  3
 *  4  5  6  7
 *  8  9 10 11
 * 12 13 14 15
 */ 
class pattern : public feature {
    public:
        // Constructor and destructor
        pattern(const std::vector<int>& p, int iso = 8) : feature(1 << (p.size() * 4)), iso_last(iso) {
            if (p.empty()) {
                error << "no pattern defined" << std::endl;
                std::exit(1);
            }
            // find the whole isomorphic patterns
            for (int i = 0; i < 8; i++) {
                board idx = 0xfedcba9876543210ull;
                if (i >= 4)
                    idx.mirror();
                idx.rotate(i);
                for (int t : p) {
                    isomorphic[i].push_back(idx.at(t));
                }
            }

        }
        pattern(const pattern& p) = delete;
        virtual ~pattern() {}
        
        // operator overloadding
        pattern& operator = (const pattern& p) = delete;

        /**
         * 估計給定盤面的價值
         */ 
        virtual float estimate(const board& b) const {
            float value = 0;
            for (int i = 0; i < iso_last; i++) {
                size_t index = indexof(isomorphic[i], b);
                value += operator[](index);
            }
            return value;
        }

        /**
         * 更新盤面上相對應位置的權重
         */
        virtual float update(const board& b, float alpha) {
            float alpha_split = alpha / iso_last;
            float value = 0;
            for (int i = 0; i < iso_last; i++) {
                size_t index = indexof(isomorphic[i], b);
                operator[](index) += alpha_split;
                value += operator[](index);
            }
            return value;
        }

        /**
         * 獲取此pattern的名子 (多少tuple)
         */
        virtual std::string name() const {
            return std::to_string(isomorphic[0].size())
                + "-tuple pattern " + nameof(isomorphic[0]);
        }

        /**
         * 設置有多少isomorphic情況
         */
        void set_isomorphic(int i = 8) {
            iso_last = i;
        }

        /**
         * 印出給定盤面的權重資訊
         */
        void dump(const board& b, std::ostream& out = info) const {
            for(int i = 0; i < iso_last; i++) {
                out << "#" << i << ":" << nameof(isomorphic[i]) << "(";
			    size_t index = indexof(isomorphic[i], b);
			    for (size_t i = 0; i < isomorphic[i].size(); i++) {
			    	out << std::hex << ((index >> (4 * i)) & 0x0f);
			    }
			    out << std::dec << ") = " << operator[](index) << std::endl;
            }
        }

        /**
         * 給定盤面和一個pattern index，輸出它在權重表中的的index
         */
        size_t indexof(const std::vector<int>& patt, const board& b) const {
            size_t index = 0;
            for (size_t i = 0; i < patt.size(); i++)
		    	index |= b.at(patt[i]) << (4 * i);
		    return index;
        }

        /**
         * 輸出pattern的index數字串，並轉成字串形式
         */
        std::string nameof(const std::vector<int>& patt) const {
            std::stringstream ss;
            ss << std::hex;
            std::copy(patt.cbegin(), patt.cend(), std::ostream_iterator<int>(ss, ""));
            return ss.str();
        }

    protected:
        std::array<std::vector<int>, 8> isomorphic;
        int iso_last;

};

/**
 * before state and after state wrapper
 */
class state {
    public:
        // Constructor
        state(int opcode = -1)
            : opcode(opcode), score(-1), esti(-std::numeric_limits<float>::max()) {}
        state(const board& b, int opcode = -1)
            : opcode(opcode), score(-1), esti(-std::numeric_limits<float>::max()) {
                assign(b);
            }
        state(const state& st) = default;

        // Operator overloadding
        state& operator = (const state& st) = default;
        bool operator == (const state& s) const {
            return (opcode == s.opcode) && (before == s.before) && (after == s.after) && (esti == s.esti) && (score == s.score);
        }
        bool operator < (const state& s) const {
    		if (before != s.before) throw std::invalid_argument("state::operator<");
    		    return esti < s.esti;
    	}
    	bool operator !=(const state& s) const { return !(*this == s); }
    	bool operator > (const state& s) const { return s < *this; }
    	bool operator <=(const state& s) const { return !(s < *this); }
    	bool operator >=(const state& s) const { return !(*this < s); }
        friend std::ostream& operator <<(std::ostream& out, const state& st) {
	    	out << "moving " << st.name() << ", reward = " << st.score;
	    	if (st.is_valid()) {
	    		out << ", value = " << st.esti << std::endl << st.after;
	    	} else {
	    		out << " (invalid)" << std::endl;
	    	}
	    	return out;
	    }   

        // 基本操作 - get and set
        board after_state()  const { return after;  }
        board before_state() const { return before; }
        float value()  const { return esti;   }
        float reward() const { return score;  }
        float action() const { return opcode; }

        void set_before_state(const board& b) { before = b; }
        void set_after_state (const board& b) { after  = b; }
        void set_value(float v) { esti = v;   }
        void set_reward (int r) { score = r;  }
        void set_action (int a) { opcode = a; }

        /**
         * Assign before state，並施加動作得到after state，
         * 回傳動作是否為無效數值
         */
        bool assign(const board& b) {
            debug << "assign " << name() << std::endl << b;
            after = before = b;
            score = after.move(opcode);
            esti = score;
            return score != -1;
        }

        /**
         * 回傳opcode代表的英文動作，
         * 如果opcode為無效值，則回傳英文字none
         */
        const char* name() const {
            static const char* opname[4] = { "up", "right", "down", "left" };
            return (opcode >= 0 && opcode < 4) ? opname[opcode] : "none";
        }

        /**
         * 檢查目前state object中的各項元素是否有效
         * 若
         * 1. value數值為NaN
         * 2. before state等於after state
         * 3. reward為-1
         * 則無效(有問題)
         */
        bool is_valid() const {
            if (std::isnan(esti)) {
                error << "numeric exception" << std::endl;
                std::exit(1);
            }
            return after != before && opcode != -1 && score != -1;
        }


    private:
        board before;
        board after;
        int opcode;
        int score;
        float esti;
};

class learning {
    public: 
        // Constructor and destructor
        learning() {}
        ~learning() {}

        /**
         * 為特定的feature配置記憶體
         */
        void add_feature(feature* feat) {
            feats.push_back(feat);

            info << "[     AI Bot    ] Allocate ";
            info << feat->name() << " , size = " << feat->size();
            size_t usage = feat->size() * sizeof(float);
            if (usage >= (1 << 30)) {
		    	info << " (" << (usage >> 30) << "GB)";
		    } else if (usage >= (1 << 20)) {
		    	info << " (" << (usage >> 20) << "MB)";
		    } else if (usage >= (1 << 10)) {
		    	info << " (" << (usage >> 10) << "KB)";
		    }
            info << std::endl;
        }

        /**
         * 載入預訓練模型
         * 注意：你需要先呼叫add_feature把所有pattern都定義好了以後，
         * 才能呼叫此函式
         */
        void load(const std::string& path) {
	    	std::ifstream in;
	    	in.open(path.c_str(), std::ios::in | std::ios::binary);
	    	if (in.is_open()) {
	    		size_t size;
	    		in.read(reinterpret_cast<char*>(&size), sizeof(size));
	    		if (size != feats.size()) {
	    			error << "unexpected feature count: " << size << " (" << feats.size() << " is expected)" << std::endl;
	    			std::exit(1);
	    		}
	    		for (feature* feat : feats) {
	    			in >> *feat;
	    			info << feat->name() << " is loaded from " << path << std::endl;
	    		}
	    		in.close();
	    	}
	    }

        /**
         * 儲存訓練模型
         */
        void save(const std::string& path) {
	    	std::ofstream out;
	    	out.open(path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
	    	if (out.is_open()) {
	    		size_t size = feats.size();
	    		out.write(reinterpret_cast<char*>(&size), sizeof(size));
	    		for (feature* feat : feats) {
	    			out << *feat;
	    			info << feat->name() << " is saved to " << path << std::endl;
	    		}
	    		out.flush();
	    		out.close();
	    	}
	    }

        /**
         * 對於一個盤面（4個pattern）估計總價值
         */
        float estimate(const board& b) const {
            debug << "estimate " << std::endl << b;
            float value = 0;
            for (feature* feat : feats) {
                value += feat->estimate(b);
            }
            return value;
        }

        /**
         * 對於一個盤面b，選擇一個最好的移動情況，並返回一個state object
         * 返回的state object中，
         * before_state() 是 t 時間點的before state，也就是s_{t}
         * after_state() 是 t 時間點的after state，也就是s'_{t}
         * action() 是 t 時間點最後做的動作，也就是a_{t}
         * reward() 是 t 時間點獲得的 reward，也就是R_{t}
         * value() 是 t 時間點透過after state value function獲得的value數值，也就是V^{af}(S^{af}_{t})
         */
        state select_best_move(const board& b) const {
            state after[4] = { 0, 1, 2, 3 };
            state* best = after;
            for (state* move = after; move != after + 4; move++) {
                if (move->assign(b)) {                                  // 如果此次移動為一個有效移動
                    move->set_value(estimate(move->after_state()));     // 設定after state value為此盤面的結果
                    // move->set_value(move->reward() + estimate(move->after_state()));
                    if (move->value() > best->value())                  // 如果有更好的after state value
                        best = move;                                    // 則更新best state
                } else {                                                // 若不為有效移動
                    move->set_value(-std::numeric_limits<float>::max());// 則設置after state value為負無限大
                }
                debug << "test " << *move;
            }
            return *best;
        }

        /**
         * 對整個episode進行更新
         * 在過程中的每一步，會呼叫update()函式來更新單一時間步
         */
        void update_episode(std::vector<state>& path, float alpha = 0.1) const {
            float exact = 0;
            for (path.pop_back(); path.size(); path.pop_back()) {
                state& move = path.back();  // 獲取要更新的state
                float error = exact - move.value();
                debug << "update error = " << error << " for after state" << std::endl << move.after_state();
                exact = move.reward() + update(move.after_state(), alpha * error);
            }
        }

        /**
         * 給定一個state更新它的權重，並返回value
         */
        float update(const board& b, float u) const {
            debug << "update " << " (" << u << ")" << std::endl << b;
            float u_split = u / feats.size();
            float value = 0;
            for (feature* feat : feats) {
                value += feat->update(b, u_split);
            }
            return value;
        }

        /**
	     * update the statistic, and display the status once in 1000 episodes by default
	     *
	     * the format would be
	     * 1000   mean = 273901  max = 382324
	     *        512     100%   (0.3%)
	     *        1024    99.7%  (0.2%)
	     *        2048    99.5%  (1.1%)
	     *        4096    98.4%  (4.7%)
	     *        8192    93.7%  (22.4%)
	     *        16384   71.3%  (71.3%)
	     *
	     * where (let unit = 1000)
	     *  '1000': current iteration (games trained)
	     *  'mean = 273901': the average score of last 1000 games is 273901
	     *  'max = 382324': the maximum score of last 1000 games is 382324
	     *  '93.7%': 93.7% (937 games) reached 8192-tiles in last 1000 games (a.k.a. win rate of 8192-tile)
	     *  '22.4%': 22.4% (224 games) terminated with 8192-tiles (the largest) in last 1000 games
	     */
	    void make_statistic(size_t n, const board& b, int score, int unit = 1000) {
	    	scores.push_back(score);
	    	maxtile.push_back(0);
	    	for (int i = 0; i < 16; i++) {
	    		maxtile.back() = std::max(maxtile.back(), b.at(i));
	    	}

	    	if (n % unit == 0) { // show the training process
	    		if (scores.size() != size_t(unit) || maxtile.size() != size_t(unit)) {
	    			error << "wrong statistic size for show statistics" << std::endl;
	    			std::exit(2);
	    		}
	    		int sum = std::accumulate(scores.begin(), scores.end(), 0);
	    		int max = *std::max_element(scores.begin(), scores.end());
	    		int stat[16] = { 0 };
	    		for (int i = 0; i < 16; i++) {
	    			stat[i] = std::count(maxtile.begin(), maxtile.end(), i);
	    		}
	    		float mean = float(sum) / unit;
	    		float coef = 100.0 / unit;
	    		info << n;
	    		info << "\t" "mean = " << mean;
	    		info << "\t" "max = " << max;
	    		info << std::endl;
	    		for (int t = 1, c = 0; c < unit; c += stat[t++]) {
	    			if (stat[t] == 0) continue;
	    			int accu = std::accumulate(stat + t, stat + 16, 0);
	    			info << "\t" << ((1 << t) & -2u) << "\t" << (accu * coef) << "%";
	    			info << "\t(" << (stat[t] * coef) << "%)" << std::endl;
	    		}
	    		scores.clear();
	    		maxtile.clear();
	    	}
	    }


    private:
        std::vector<feature*> feats;    // 紀錄有哪些feature的表
        std::vector<int> scores;
	    std::vector<int> maxtile;
};

int main() {
    info << "<< TD learning for game 2048 >>" << std::endl;
    learning tdl;

    // Set the learning parameters
    float alpha  = 0.1;         // 學習率
    size_t total = 10000;       // 遊玩次數
    unsigned seed;              // 隨機種子
    __asm__ __volatile__ ("rdtsc" : "=a" (seed));
	info << "[hyper-parameter] alpha = " << alpha << std::endl;
	info << "[hyper-parameter] total = " << total << std::endl;
	info << "[hyper-parameter] seed  = " << seed << std::endl;
    std::srand(seed);

    // Initialize the features
    // tdl.add_feature(new pattern({0, 1, 2, 3}));
    // tdl.add_feature(new pattern({4, 5, 6, 7}));
    tdl.add_feature(new pattern({ 0, 1, 2, 4, 5, 6 }));
	tdl.add_feature(new pattern({ 4, 5, 6, 8, 9, 10 }));

    // Load the pre-trained model
    tdl.load("");

    // Train
    std::vector<state> path;    // 紀錄每一個時間點的state
    path.reserve(20000);        // 為vector保留至少20000個state空間
    for(size_t episode = 1; episode <= total; episode++) {
        // 初始化盤面和total revard
        debug << "begin episode" << std::endl;
        board b;
        b.init();
        int total_reward = 0;
        while (true) {
        // for(size_t i = 0; i < 1; i++) {
            debug << "state" << std::endl << b;
            state best = tdl.select_best_move(b);
            path.push_back(best);

            if (best.is_valid()) {                  // 如果best state是有效的state
                debug << "best " << best;
                total_reward += best.reward();      // 累加return
                b = best.after_state();
                b.popup();                          // 那就擷取best的after state，並跳出亂數tile
            } else {                                // 如果best state是無效的state則結束遊戲
                break;
            }
        }
        debug << "end episode" << std::endl;

        // 更新 - TD afterstates
        tdl.update_episode(path, alpha);
        tdl.make_statistic(episode, b, total_reward);
        path.clear();
    }

    return 0;
}