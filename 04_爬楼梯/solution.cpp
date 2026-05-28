/**
 * 70. 爬楼梯
 * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
 * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
 */

// 解法1：动态规划 - 空间优化版（推荐）
class Solution {
public:
    int climbStairs(int n) {
        if(n == 1) return 1;
        if(n == 2) return 2;

        int prev2 = 1;  // f(1) = 1
        int prev1 = 2;  // f(2) = 2
        int curr;

        for(int i = 3; i <= n; i++) {
            curr = prev1 + prev2;  // f(i) = f(i-1) + f(i-2)
            prev2 = prev1;         // 滚动更新
            prev1 = curr;
        }

        return curr;
    }
};

// 解法2：动态规划 - 数组版（更易理解）
class Solution2 {
public:
    int climbStairs(int n) {
        if(n == 1) return 1;
        if(n == 2) return 2;

        vector<int> dp(n + 1);
        dp[1] = 1;
        dp[2] = 2;

        for(int i = 3; i <= n; i++) {
            dp[i] = dp[i-1] + dp[i-2];
        }

        return dp[n];
    }
};

// 解法3：递归 + 记忆化（避免重复计算）
class Solution3 {
public:
    int climbStairs(int n) {
        vector<int> memo(n + 1, -1);
        return helper(n, memo);
    }

private:
    int helper(int n, vector<int>& memo) {
        if(n == 1) return 1;
        if(n == 2) return 2;

        if(memo[n] != -1) return memo[n];

        memo[n] = helper(n-1, memo) + helper(n-2, memo);
        return memo[n];
    }
};

// 解法4：纯递归（会超时，仅用于理解）
class Solution4 {
public:
    int climbStairs(int n) {
        if(n == 1) return 1;
        if(n == 2) return 2;
        return climbStairs(n-1) + climbStairs(n-2);
    }
};
