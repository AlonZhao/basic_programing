#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// 解法一：交换 + 冒泡（完全原地）
void rearrange_bubble(vector<int>& X, vector<int>& Y) {
    int m = X.size();
    int n = Y.size();

    int i = m - 1;  // X的最大元素位置
    int j = n - 1;  // Y的最小元素位置

    while (i >= 0 && j >= 0) {
        if (X[i] > Y[j]) {
            swap(X[i], Y[j]);

            // X[i]变小了，向前冒泡保持升序
            int k = i;
            while (k > 0 && X[k] < X[k-1]) {
                swap(X[k], X[k-1]);
                k--;
            }

            // Y[j]变大了，向前冒泡保持降序
            k = j;
            while (k > 0 && Y[k] > Y[k-1]) {
                swap(Y[k], Y[k-1]);
                k--;
            }
        }

        i--;
        j--;
    }
}

// 解法二：交换 + 排序（推荐）
void rearrange(vector<int>& X, vector<int>& Y) {
    int m = X.size();
    int n = Y.size();

    int i = m - 1;
    int j = n - 1;

    // 从尾部开始交换所有不符合条件的元素
    while (i >= 0 && j >= 0 && X[i] > Y[j]) {
        swap(X[i], Y[j]);
        i--;
        j--;
    }

    // 重新排序
    sort(X.begin(), X.end());               // 升序
    sort(Y.begin(), Y.end(), greater<int>());  // 降序
}

// 打印数组
void print_array(const string& name, const vector<int>& arr) {
    cout << name << " = [";
    for (int i = 0; i < arr.size(); i++) {
        cout << arr[i];
        if (i < arr.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
}

int main() {
    // 测试用例1
    vector<int> X1 = {1, 3, 5, 7};
    vector<int> Y1 = {8, 6, 4, 2};

    cout << "测试用例1：" << endl;
    cout << "输入：" << endl;
    print_array("X", X1);
    print_array("Y", Y1);

    rearrange(X1, Y1);

    cout << "输出：" << endl;
    print_array("X", X1);
    print_array("Y", Y1);
    cout << endl;

    // 测试用例2：X和Y大小不同
    vector<int> X2 = {1, 5, 9};
    vector<int> Y2 = {10, 8, 6, 4, 2};

    cout << "测试用例2：" << endl;
    cout << "输入：" << endl;
    print_array("X", X2);
    print_array("Y", Y2);

    rearrange(X2, Y2);

    cout << "输出：" << endl;
    print_array("X", X2);
    print_array("Y", Y2);
    cout << endl;

    // 测试用例3：已经满足条件
    vector<int> X3 = {1, 2, 3};
    vector<int> Y3 = {6, 5, 4};

    cout << "测试用例3（已满足条件）：" << endl;
    cout << "输入：" << endl;
    print_array("X", X3);
    print_array("Y", Y3);

    rearrange(X3, Y3);

    cout << "输出：" << endl;
    print_array("X", X3);
    print_array("Y", Y3);
    cout << endl;

    // 测试用例4：完全需要交换
    vector<int> X4 = {5, 6, 7};
    vector<int> Y4 = {4, 3, 2, 1};

    cout << "测试用例4（完全交换）：" << endl;
    cout << "输入：" << endl;
    print_array("X", X4);
    print_array("Y", Y4);

    rearrange(X4, Y4);

    cout << "输出：" << endl;
    print_array("X", X4);
    print_array("Y", Y4);

    return 0;
}
