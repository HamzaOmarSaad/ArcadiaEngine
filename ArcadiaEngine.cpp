// ArcadiaEngine.cpp - STUDENT TEMPLATE
// TODO: Implement all the functions below according to the assignment requirements

#include "ArcadiaEngine.h"
#include <algorithm>
#include <queue>
#include <numeric>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <set>
#include <optional>
#include <random>
#include <chrono>

using namespace std;

// =========================================================
// PART A: DATA STRUCTURES (Concrete Implementations)
// =========================================================

struct PlayerData {
  int PlayerID;
  string Name;
};

enum class SlotState {
  Empty,
  Occupied,
  Deleted
};

const int TableSize = 101;
const int R = 97;
const double A = 0.6180339887;

// --- 1. PlayerTable (Double Hashing) ---

class ConcretePlayerTable : public PlayerTable {
private:
    vector<optional<PlayerData>> table;
    vector<SlotState> state;

    int H1(int key) const {
        double frac = key * A;
        frac = frac - floor(frac);
        return (int)(frac * TableSize);
    }

    int H2(int key) const {
        return R - (key % R);
    }

public:
    ConcretePlayerTable() {
        table.resize(TableSize);
        state.resize(TableSize, SlotState::Empty);
    }

    void insert(int playerID, string name) override {
        int index1 = H1(playerID);
        int step = H2(playerID);

        for (int i = 0; i < TableSize; i++) {
            int idx = (index1 + i * step) % TableSize;

            if (state[idx] == SlotState::Empty || state[idx] == SlotState::Deleted) {
                table[idx] = PlayerData{playerID, name};
                state[idx] = SlotState::Occupied;
                return;
            }

            if (state[idx] == SlotState::Occupied &&
                table[idx]->PlayerID == playerID) {
                table[idx] = PlayerData{playerID, name}; // update
                return;
            }
        }

        cout << "Error: Table is full\n";
    }

    string search(int playerID) override {
        int index1 = H1(playerID);
        int step = H2(playerID);

        for (int i = 0; i < TableSize; i++) {
            int idx = (index1 + i * step) % TableSize;

            if (state[idx] == SlotState::Empty)
                return "Player Not Found\n";

            if (state[idx] == SlotState::Occupied &&
                table[idx]->PlayerID == playerID)
                return table[idx]->Name;
        }

        return "Player Not Found\n";
    }
};


const int MaxLevel = 16;
const double P = 0.5;

struct ScoreNode {
  int PlayerID;
  int score;
  vector<ScoreNode*> forward;
  ScoreNode(int id, int scr, int level) : PlayerID(id), score(scr) {
    forward.resize(level + 1, nullptr);
  }
};

// --- 2. Leaderboard (Skip List) ---

class ConcreteLeaderboard : public Leaderboard {
private:
    ScoreNode* head;
    int currentLevel;
    default_random_engine gen;

    int randomLevel() {
        int lvl = 0;
        uniform_real_distribution<double> dist(0.0, 1.0);
        while (dist(gen) < P && lvl < MaxLevel - 1)
            lvl++;
        return lvl;
    }

    // Order: score DESC, PlayerID ASC
    bool goesBefore(int id1, int score1, int id2, int score2) {
        if (score1 != score2)
            return score1 > score2;
        return id1 < id2;
    }

public:
    ConcreteLeaderboard() : currentLevel(0) {
        gen.seed(chrono::steady_clock::now().time_since_epoch().count());
        head = new ScoreNode(-1, INT_MAX, MaxLevel);
    }

    ~ConcreteLeaderboard() {
        ScoreNode* cur = head->forward[0];
        while (cur) {
            ScoreNode* nxt = cur->forward[0];
            delete cur;
            cur = nxt;
        }
        delete head;
    }

    void addScore(int playerID, int score) override {
        vector<ScoreNode*> update(MaxLevel + 1);
        ScoreNode* cur = head;

        for (int i = currentLevel; i >= 0; i--) {
            while (cur->forward[i] &&
                   goesBefore(cur->forward[i]->PlayerID,
                              cur->forward[i]->score,
                              playerID, score)) {
                cur = cur->forward[i];
            }
            update[i] = cur;
        }

        int lvl = randomLevel();
        if (lvl > currentLevel) {
            for (int i = currentLevel + 1; i <= lvl; i++)
                update[i] = head;
            currentLevel = lvl;
        }

        ScoreNode* node = new ScoreNode(playerID, score, lvl);
        for (int i = 0; i <= lvl; i++) {
            node->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = node;
        }
    }

    void removePlayer(int playerID) override {
        // Allowed O(N) scan
        ScoreNode* cur = head->forward[0];
        int score = -1;

        while (cur) {
            if (cur->PlayerID == playerID) {
                score = cur->score;
                break;
            }
            cur = cur->forward[0];
        }

        if (score == -1)
            return;

        vector<ScoreNode*> update(MaxLevel + 1);
        cur = head;

        for (int i = currentLevel; i >= 0; i--) {
            while (cur->forward[i] &&
                   goesBefore(cur->forward[i]->PlayerID,
                              cur->forward[i]->score,
                              playerID, score)) {
                cur = cur->forward[i];
            }
            update[i] = cur;
        }

        ScoreNode* target = cur->forward[0];
        if (!target || target->PlayerID != playerID)
            return;

        for (int i = 0; i <= currentLevel; i++) {
            if (update[i]->forward[i] == target)
                update[i]->forward[i] = target->forward[i];
        }

        delete target;

        while (currentLevel > 0 && head->forward[currentLevel] == nullptr)
            currentLevel--;
    }

    vector<int> getTopN(int n) override {
        vector<int> result;
        ScoreNode* cur = head->forward[0];

        while (cur && n--) {
            result.push_back(cur->PlayerID);
            cur = cur->forward[0];
        }
        return result;
    }
};

// --- 3. AuctionTree (Red-Black Tree) ---

enum class Color {
  RED,
  BLACK
};

struct ItemNode {
    int itemID;
    int price;
    Color color;
    ItemNode *left, *right, *parent;

    ItemNode(int id = -1, int p = 0, Color c = Color::BLACK)
        : itemID(id), price(p), color(c),
          left(nullptr), right(nullptr), parent(nullptr) {}
};

class ConcreteAuctionTree : public AuctionTree {
private:
    ItemNode* root;
    ItemNode* NIL;

    bool lessThan(int price1, int id1, int price2, int id2) const {
        if (price1 != price2) return price1 < price2;
        return id1 < id2;
    }

    void leftRotate(ItemNode* x) {
        ItemNode* y = x->right;
        x->right = y->left;
        if (y->left != NIL) y->left->parent = x;

        y->parent = x->parent;
        if (x->parent == NIL) root = y;
        else if (x == x->parent->left) x->parent->left = y;
        else x->parent->right = y;

        y->left = x;
        x->parent = y;
    }

    void rightRotate(ItemNode* y) {
        ItemNode* x = y->left;
        y->left = x->right;
        if (x->right != NIL) x->right->parent = y;

        x->parent = y->parent;
        if (y->parent == NIL) root = x;
        else if (y == y->parent->left) y->parent->left = x;
        else y->parent->right = x;

        x->right = y;
        y->parent = x;
    }

    void fixInsert(ItemNode* z) {
        while (z->parent->color == Color::RED) {
            if (z->parent == z->parent->parent->left) {
                ItemNode* y = z->parent->parent->right;
                if (y->color == Color::RED) {
                    z->parent->color = Color::BLACK;
                    y->color = Color::BLACK;
                    z->parent->parent->color = Color::RED;
                    z = z->parent->parent;
                } else {
                    if (z == z->parent->right) {
                        z = z->parent;
                        leftRotate(z);
                    }
                    z->parent->color = Color::BLACK;
                    z->parent->parent->color = Color::RED;
                    rightRotate(z->parent->parent);
                }
            } else {
                ItemNode* y = z->parent->parent->left;
                if (y->color == Color::RED) {
                    z->parent->color = Color::BLACK;
                    y->color = Color::BLACK;
                    z->parent->parent->color = Color::RED;
                    z = z->parent->parent;
                } else {
                    if (z == z->parent->left) {
                        z = z->parent;
                        rightRotate(z);
                    }
                    z->parent->color = Color::BLACK;
                    z->parent->parent->color = Color::RED;
                    leftRotate(z->parent->parent);
                }
            }
        }
        root->color = Color::BLACK;
    }

    void transplant(ItemNode* u, ItemNode* v) {
        if (u->parent == NIL) root = v;
        else if (u == u->parent->left) u->parent->left = v;
        else u->parent->right = v;
        v->parent = u->parent;
    }

    ItemNode* treeMinimum(ItemNode* x) {
        while (x->left != NIL)
            x = x->left;
        return x;
    }

    void fixDelete(ItemNode* x) {
        while (x != root && x->color == Color::BLACK) {
            if (x == x->parent->left) {
                ItemNode* w = x->parent->right;
                if (w->color == Color::RED) {
                    w->color = Color::BLACK;
                    x->parent->color = Color::RED;
                    leftRotate(x->parent);
                    w = x->parent->right;
                }
                if (w->left->color == Color::BLACK &&
                    w->right->color == Color::BLACK) {
                    w->color = Color::RED;
                    x = x->parent;
                } else {
                    if (w->right->color == Color::BLACK) {
                        w->left->color = Color::BLACK;
                        w->color = Color::RED;
                        rightRotate(w);
                        w = x->parent->right;
                    }
                    w->color = x->parent->color;
                    x->parent->color = Color::BLACK;
                    w->right->color = Color::BLACK;
                    leftRotate(x->parent);
                    x = root;
                }
            } else {
                ItemNode* w = x->parent->left;
                if (w->color == Color::RED) {
                    w->color = Color::BLACK;
                    x->parent->color = Color::RED;
                    rightRotate(x->parent);
                    w = x->parent->left;
                }
                if (w->right->color == Color::BLACK &&
                    w->left->color == Color::BLACK) {
                    w->color = Color::RED;
                    x = x->parent;
                } else {
                    if (w->left->color == Color::BLACK) {
                        w->right->color = Color::BLACK;
                        w->color = Color::RED;
                        leftRotate(w);
                        w = x->parent->left;
                    }
                    w->color = x->parent->color;
                    x->parent->color = Color::BLACK;
                    w->left->color = Color::BLACK;
                    rightRotate(x->parent);
                    x = root;
                }
            }
        }
        x->color = Color::BLACK;
    }

    ItemNode* findNodeByID_ON(int itemID) {
        vector<ItemNode*> stack;
        if (root != NIL) stack.push_back(root);

        while (!stack.empty()) {
            ItemNode* cur = stack.back();
            stack.pop_back();

            if (cur->itemID == itemID)
                return cur;

            if (cur->right != NIL) stack.push_back(cur->right);
            if (cur->left != NIL) stack.push_back(cur->left);
        }
        return NIL;
    }

public:
    ConcreteAuctionTree() {
        NIL = new ItemNode();
        NIL->left = NIL;
        NIL->right = NIL;
        NIL->parent = NIL;
        NIL->color = Color::BLACK;
        root = NIL;
    }

    ~ConcreteAuctionTree() {
        vector<ItemNode*> stack;
        if (root != NIL) stack.push_back(root);
        while (!stack.empty()) {
            ItemNode* cur = stack.back();
            stack.pop_back();
            if (cur->left != NIL) stack.push_back(cur->left);
            if (cur->right != NIL) stack.push_back(cur->right);
            delete cur;
        }
        delete NIL;
    }

    void insertItem(int itemID, int price) override {
        ItemNode* z = new ItemNode(itemID, price, Color::RED);
        z->left = z->right = z->parent = NIL;

        ItemNode* y = NIL;
        ItemNode* x = root;

        while (x != NIL) {
            y = x;
            if (lessThan(z->price, z->itemID, x->price, x->itemID))
                x = x->left;
            else
                x = x->right;
        }

        z->parent = y;
        if (y == NIL) root = z;
        else if (lessThan(z->price, z->itemID, y->price, y->itemID))
            y->left = z;
        else
            y->right = z;

        fixInsert(z);
    }

    void deleteItem(int itemID) override {
        ItemNode* z = findNodeByID_ON(itemID);
        if (z == NIL) return;

        ItemNode* y = z;
        ItemNode* x;
        Color yColor = y->color;

        if (z->left == NIL) {
            x = z->right;
            transplant(z, z->right);
        } else if (z->right == NIL) {
            x = z->left;
            transplant(z, z->left);
        } else {
            y = treeMinimum(z->right);
            yColor = y->color;
            x = y->right;

            if (y->parent == z)
                x->parent = y;
            else {
                transplant(y, y->right);
                y->right = z->right;
                y->right->parent = y;
            }

            transplant(z, y);
            y->left = z->left;
            y->left->parent = y;
            y->color = z->color;
        }

        delete z;

        if (yColor == Color::BLACK)
            fixDelete(x);
    }
};

// =========================================================
// PART B: INVENTORY SYSTEM (Dynamic Programming)
// =========================================================



int InventorySystem::optimizeLootSplit(int n, vector<int>& coins) {
    if (n == 0) {
        return 0;
    }


    int totalSum = accumulate(coins.begin(), coins.end(), 0);


    int targetSum = totalSum / 2;


    vector<bool> dp(targetSum + 1, false);
    dp[0] = true;


    for (int coin : coins) {

        for (int j = targetSum; j >= coin; --j) {
            dp[j] = dp[j] || dp[j - coin];
        }
    }


    int sum1 = 0;
    for (int j = targetSum; j >= 0; --j) {
        if (dp[j]) {
            sum1 = j;
            break;
        }
    }


    int sum2 = totalSum - sum1;

    return sum2 - sum1;
}

int InventorySystem::maximizeCarryValue(int capacity, vector<pair<int, int>>& items) {
    if (capacity == 0 || items.empty()) {
        return 0;
    }


    vector<int> dp(capacity + 1, 0);

    for (const auto& item : items) {
        int weight = item.first;
        int value = item.second;


        for (int w = capacity; w >= weight; --w) {

            int value_exclude = dp[w];


            int value_include = value + dp[w - weight];


            dp[w] = max(value_exclude, value_include);
        }
    }


    return dp[capacity];
}

long long InventorySystem::countStringPossibilities(string s) {
    int N = s.length();
    if (N == 0) {
        return 1;
    }


    long long MOD = 1e9 + 7;


    vector<long long> dp(N + 1, 0);


    dp[0] = 1;


    for (int i = 1; i <= N; ++i) {

        dp[i] = dp[i-1];


        if (i >= 2) {

            char c1 = s[i-2];
            char c2 = s[i-1];


            if (c1 == 'u' && c2 == 'u') {

                dp[i] = (dp[i] + dp[i-2]) % MOD;
            }

            else if (c1 == 'n' && c2 == 'n') {

                dp[i] = (dp[i] + dp[i-2]) % MOD;
            }
        }

        dp[i] = dp[i] % MOD;
    }

    return dp[N];
}

// =========================================================
// PART C: WORLD NAVIGATOR (Graphs)
// =========================================================

bool WorldNavigator::pathExists(int n, vector<vector<int>>& edges, int source, int dest) {
    //if source equals destination
    if (source == dest) return true;

    // Step 1: Create and fill adjacency matrix (n x n)
    vector<vector<int>> matrix(n, vector<int>(n, 0));

    for (const auto& edge : edges) {
        int u = edge[0];
        int v = edge[1];
        matrix[u][v] = 1;
        matrix[v][u] = 1;  // Bidirectional
    }

    // Step 2: BFS to check connectivity
    vector<bool> visited(n, false);
    queue<int> q;

    visited[source] = true;
    q.push(source);

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        // If we reached destination, path exists
        if (current == dest) return true;

        // Check all cities to see if they're neighbors of current
        for (int neighbor = 0; neighbor < n; neighbor++) {
            // If there's a connection AND we haven't visited it yet
            if (matrix[current][neighbor] == 1 && !visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }

    // Destination never reached
    return false;
}

long long WorldNavigator::minBribeCost(int n, int m, long long goldRate, long long silverRate,
                                       vector<vector<int>>& roadData) {
    // Convert costs to Tugriks
    vector<vector<long long>> edges(m, vector<long long>(3));
    for (int i = 0; i < m; i++) {
        edges[i][0] = roadData[i][0];  // u
        edges[i][1] = roadData[i][1];  // v
        edges[i][2] = roadData[i][2] * goldRate + roadData[i][3] * silverRate; // cost
    }

    vector<bool> selected(m, false);
    vector<int> parent(n);

    // Initialize Union-Find
    for (int i = 0; i < n; i++) parent[i] = i;

    long long minCost = 0;
    int edgeCount = 0;

    while (edgeCount < n - 1) {
        long long min = 1e18;
        int index = -1;

        // Find cheapest unselected edge
        for (int i = 0; i < m; i++) {
            if (!selected[i] && edges[i][2] < min) {
                min = edges[i][2];
                index = i;
            }
        }

        if (index == -1) break; // No more edges

        selected[index] = true;
        int u = edges[index][0];
        int v = edges[index][1];

        // Find parents (Union-Find check)
        int parentU = u;
        while (parent[parentU] != parentU) parentU = parent[parentU];

        int parentV = v;
        while (parent[parentV] != parentV) parentV = parent[parentV];

        // If different components, add edge
        if (parentU != parentV) {
            minCost += edges[index][2];
            edgeCount++;
            parent[parentU] = parentV; // Union
        }
    }

    // Check if all cities connected
    if (edgeCount != n - 1) return -1;
    return minCost;
}

string WorldNavigator::sumMinDistancesBinary(int n, vector<vector<int>>& roads) {
    const long long INF = 1e18;
    // 1. Initialize distance matrix
    vector<vector<long long>> dist(n, vector<long long>(n, INF));
    for(int i = 0; i < n; i++) {
        dist[i][i] = 0;
    }

    // 2. Add  edges
    for(auto& road : roads) {
        int u = road[0];
        int v = road[1];
        int w = road[2];
        dist[u][v] = w;
    }

    // 3. Floyd-Warshall for all pairs shortest path
    for(int k = 0; k < n; k++) {
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                if(dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    // 4. Sum distances from smaller to larger index
    long long total = 0;
    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            if(dist[i][j] < INF) {
                total += dist[i][j];
            }
        }
    }

    // 5. Convert total to binary string
    if(total == 0) return "0";

    string binary = "";
    while(total > 0) {
        binary = to_string(total % 2) + binary;
        total /= 2;
    }
    return binary;
}

// =========================================================
// PART D: SERVER KERNEL (Greedy)
// =========================================================

int ServerKernel::minIntervals(vector<char>& tasks, int n) {
    int freq[26] = {0};

    for(char t : tasks) {
        if(t >= 'a' && t <= 'z') {
            freq[t - 'a']++;
        }
    }
    int maxFreq = 0;
    for(int f : freq)
        maxFreq = max(maxFreq, f);

    int countMax = 0;
    for(int f : freq)
        if(f == maxFreq)
            countMax++;

    int part1 = (maxFreq - 1) * (n + 1) + countMax;

    return max((int)tasks.size(), part1);

}

// =========================================================
// FACTORY FUNCTIONS (Required for Testing)
// =========================================================

extern "C" {
    PlayerTable* createPlayerTable() {
        return new ConcretePlayerTable();
    }

    Leaderboard* createLeaderboard() {
        return new ConcreteLeaderboard();
    }

    AuctionTree* createAuctionTree() {
        return new ConcreteAuctionTree();
    }
}