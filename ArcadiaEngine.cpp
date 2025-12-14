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

using namespace std;

// =========================================================
// PART A: DATA STRUCTURES (Concrete Implementations)
// =========================================================

struct PlayerData {
  int PlayerID;
  string Name;
}

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

  int H1Multiplication(int key) const {
    double fracPart = (double)key * A
    fracPart = fracPart - floor(fracPart);
    return (int)floor(fracPart * TableSize);
  }

  int H2Step (int key) const {
    return R - (key % R);
  }

public:
    ConcretePlayerTable() {
        table.resize(TableSize);
        state.resize(TableSize, slotState::Empty);
    }

    void insert(int playerID, string name) override {
        PlayerData playerData = {playerID, name};
        int key = playerData.PlayerID;
        int initialIndex = H1Multiplication(key);
        int step = H2Step(key);
        for (int i = 0; i < table.size(); i++) {
          int index = (initialIndex + i * step) % TableSize;
          if (state[index] == SlotState::Empty || state[index] == SlotState::Deleted) {
            table[index] = playerData;
            state[index] = SlotState::Occupied;
            return;
          }
          if (state[index] == SlotState::Occupied && table[index]->PlayerID == key) {
            table[index] = playerData;
            return;
          }
        }
    }

    string search(int playerID) override {
        int key = playerData.PlayerID;
        int initialIndex = H1Multiplication(key);
        int step = H2Step(key);
        for (int i = 0; i < table.size(); i++) {
          int index = (initialIndex + i * step) % TableSize;
          if (state[index] == SlotState::Empty) {
            return;
          }
          if (state[index] == SlotState::Occupied && table[index]->PlayerID == key) {
            return table[index]->Name;
          }
        }
        return "Player Not Found\n";
    }
};

// --- 2. Leaderboard (Skip List) ---

class ConcreteLeaderboard : public Leaderboard {
private:
    // TODO: Define your skip list node structure and necessary variables
    // Hint: You'll need nodes with multiple forward pointers

public:
    ConcreteLeaderboard() {
        // TODO: Initialize your skip list
    }

    void addScore(int playerID, int score) override {
        // TODO: Implement skip list insertion
        // Remember to maintain descending order by score
    }

    void removePlayer(int playerID) override {
        // TODO: Implement skip list deletion
    }

    vector<int> getTopN(int n) override {
        // TODO: Return top N player IDs in descending score order
        return {};
    }
};

// --- 3. AuctionTree (Red-Black Tree) ---

class ConcreteAuctionTree : public AuctionTree {
private:
    // TODO: Define your Red-Black Tree node structure
    // Hint: Each node needs: id, price, color, left, right, parent pointers

public:
    ConcreteAuctionTree() {
        // TODO: Initialize your Red-Black Tree
    }

    void insertItem(int itemID, int price) override {
        // TODO: Implement Red-Black Tree insertion
        // Remember to maintain RB-Tree properties with rotations and recoloring
    }

    void deleteItem(int itemID) override {
        // TODO: Implement Red-Black Tree deletion
        // This is complex - handle all cases carefully
    }
};

// =========================================================
// PART B: INVENTORY SYSTEM (Dynamic Programming)
// =========================================================

int InventorySystem::optimizeLootSplit(int n, vector<int>& coins) {
    // TODO: Implement partition problem using DP
    // Goal: Minimize |sum(subset1) - sum(subset2)|
    // Hint: Use subset sum DP to find closest sum to total/2
    return 0;
}

int InventorySystem::maximizeCarryValue(int capacity, vector<pair<int, int>>& items) {
    // TODO: Implement 0/1 Knapsack using DP
    // items = {weight, value} pairs
    // Return maximum value achievable within capacity
    return 0;
}

long long InventorySystem::countStringPossibilities(string s) {
    // TODO: Implement string decoding DP
    // Rules: "uu" can be decoded as "w" or "uu"
    //        "nn" can be decoded as "m" or "nn"
    // Count total possible decodings
    return 0;
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