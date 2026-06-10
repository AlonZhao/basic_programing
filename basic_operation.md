# 基本操作速查

刷题常用的数据结构基本操作整理，便于快速回忆 API 与写法。

## 1. vector 增删查改

```cpp
#include <vector>
#include <algorithm>  // find 等算法必须包含这个头文件
using namespace std;

vector<int> arr;
```

### 增

```cpp
arr.push_back(i);                  // 尾部追加
arr.insert(arr.begin() + 1, 66);   // 在下标 1 处插入 66，其后元素右移
```

### 删

```cpp
arr.pop_back();                    // 删除尾部元素
arr.erase(arr.begin() + 1);        // 删除下标 1 的元素，其后元素左移
```

### 查

```cpp
auto it = find(arr.begin(), arr.end(), 666);
if (it == arr.end()) {
    // 没找到
} else {
    int idx = it - arr.begin();    // 找到，计算下标
}
```

### 改

```cpp
arr[1] = 100;                      // 不做越界检查，速度快
arr.at(1) = 100;                   // 越界抛 out_of_range 异常，更安全
```

**复杂度提示：**
- `push_back` / `pop_back`：均摊 O(1)
- 中间 `insert` / `erase`：O(n)，因为要搬移后续元素
- 下标访问 `arr[i]`：O(1)
- `find`：O(n)，顺序查找

---

## 2. 链表节点

```cpp
class ListNode {
public:
    int val;
    ListNode* next;
    ListNode* prev;

    // 单参构造：next、prev 置空
    ListNode(int x) : val(x), next(nullptr), prev(nullptr) {}

    // 三参构造：同时指定前后指针
    ListNode(int x, ListNode* prev, ListNode* next) {
        this->val = x;
        this->prev = prev;   // 参数与成员同名，必须用 this-> 区分
        this->next = next;
    }
};
```



**易错点：**
- 空指针用 `nullptr`，不要用 `null` 或 `NULL`（C++ 推荐 `nullptr`）。
- 当构造函数参数名与成员名相同时（如 `prev`、`next`），赋值时必须用 `this->` 指明左边是成员，否则会变成参数自己给自己赋值，成员不会被初始化。
- 初始化列表（`: val(x), next(nullptr)`）里 `成员(参数)` 的写法即使同名也能区分，且效率优于在函数体内赋值。

### 创建节点：堆分配 vs 栈分配

```cpp
ListNode* head = new ListNode(arr[0]);  // 堆分配，链表场景用这个
ListNode  node(arr[0]);                 // 栈分配，出作用域即销毁
```

| 写法 | 分配位置 | 生命周期 | 访问成员 | 适用场景 |
|------|---------|---------|---------|---------|
| `new ListNode(x)` | 堆 | 手动 `delete` 释放 | `->` | 链表、需跨函数存活 |
| `ListNode node(x)` | 栈 | 出作用域自动销毁 | `.` | 临时局部对象 |

**为什么链表节点必须用 `new`：**
- 链表节点要靠 `next` 指针互相串联，并且通常要跨函数返回（如返回链表头）。
- 栈对象出了作用域就销毁，返回它的地址会得到悬空指针（undefined behavior）：

```cpp
ListNode* build() {
    ListNode head(0);   // 栈上
    return &head;       // 错误：函数返回后 head 已销毁，指针悬空
}
```

- 用 `new` 在堆上分配，节点生命周期不受函数作用域限制，因此 LeetCode 链表题清一色用 `new`。
- 代价：堆对象不会自动回收，用完需 `delete`，否则内存泄漏。
- 现代 C++ 可用 `std::make_unique<ListNode>(x)` 自动释放，但需把 `next` 也改成智能指针，刷题一般直接用裸指针 `new`。

### 易错：构造函数里变量遮蔽（shadowing）成员

写双向链表（如 LeetCode 707）时踩过的坑：在构造函数里给成员指针赋值，结果又写了一遍类型。

```cpp
class MyLinkedList {
    ListNode *fake_head;
    ListNode *fake_tail;

    MyLinkedList() {
        ListNode *fake_head = new ListNode();  // ❌ 这是新建局部变量，不是给成员赋值
        ListNode *fake_tail = new ListNode();
        fake_head->next = fake_tail;           // 只改了局部变量
        fake_tail->prev = fake_head;
    }  // 函数结束，局部变量销毁 + 内存泄漏；成员仍是野指针
};
```

**后果：** 成员 `fake_head` / `fake_tail` 从没被赋值，一直是野指针。后续任何 `fake_head->next` 都会崩，UBSan 报错：

```
runtime error: member access within misaligned address 0xbebebebebebebebe
for type 'ListNode', which requires 8 byte alignment
```

> `0xbebebebebebebebe` 是 sanitizer 给**未初始化内存**填充的特征字节（`0xBE` 重复 8 次），看到它基本就是「解引用了没初始化的指针」。

**正确写法：去掉类型，直接给成员赋值。**

```cpp
MyLinkedList() {
    fake_head = new ListNode();   // ✅ 没有 ListNode*，赋值给成员
    fake_tail = new ListNode();
    fake_head->next = fake_tail;
    fake_tail->prev = fake_head;
    size = 0;
}
```

**一句话区分：**

```cpp
ListNode *fake_head = new ListNode();  // 声明一个新变量（局部，遮蔽成员）
fake_head = new ListNode();            // 给已存在的成员变量赋值
```

**防御技巧：** 成员用 in-class 初始化置空，即使忘了赋值也是 `nullptr`（容易定位），不会是 `0xbebe...`：

```cpp
ListNode *fake_head = nullptr;
ListNode *fake_tail = nullptr;
int size = 0;
```

---

## 3. 用双链表实现栈和队列

栈和队列都只需要在**两端**操作,双链表天然支持 O(1) 的头尾插入/删除。关键区别:**栈只在一端操作,队列一端进另一端出**。

### 核心映射

| 数据结构 | 插入位置 | 删除位置 | 复杂度 |
|---------|---------|---------|--------|
| **栈(Stack)** | 头部(栈顶) | 头部 | O(1) |
| **队列(Queue)** | 尾部(队尾) | 头部(队首) | O(1) |

**为什么用双链表?** 单链表删除尾节点需要遍历找前驱 → O(n),双链表有 `prev` 指针,头尾删除都是 O(1)。

### 3.1 栈(Stack)实现

栈是**后进先出(LIFO)**,只在**头部**操作:push = 头部插入,pop = 头部删除。

```cpp
class MyStack {
private:
    struct ListNode {
        int val;
        ListNode *prev, *next;
        ListNode(int x = 0) : val(x), prev(nullptr), next(nullptr) {}
    };
    
    ListNode *fake_head = nullptr;
    ListNode *fake_tail = nullptr;
    int size = 0;

    void addBefore(ListNode *cur, int val) {
        ListNode *node = new ListNode(val);
        node->prev = cur->prev;
        node->next = cur;
        cur->prev->next = node;
        cur->prev = node;
        size++;
    }

public:
    MyStack() {
        fake_head = new ListNode();
        fake_tail = new ListNode();
        fake_head->next = fake_tail;
        fake_tail->prev = fake_head;
    }

    void push(int x) {
        addBefore(fake_head->next, x);  // 在头部插入(栈顶)
    }

    int pop() {
        if (empty()) return -1;
        ListNode *top = fake_head->next;
        int val = top->val;
        // 摘除栈顶:前后节点互相接回
        top->prev->next = top->next;
        top->next->prev = top->prev;
        delete top;
        size--;
        return val;
    }

    int top() {
        if (empty()) return -1;
        return fake_head->next->val;
    }

    bool empty() {
        return size == 0;
    }
};
```

### 3.2 队列(Queue)实现

队列是**先进先出(FIFO)**,**尾部入队,头部出队**。

```cpp
class MyQueue {
private:
    struct ListNode {
        int val;
        ListNode *prev, *next;
        ListNode(int x = 0) : val(x), prev(nullptr), next(nullptr) {}
    };
    
    ListNode *fake_head = nullptr;
    ListNode *fake_tail = nullptr;
    int size = 0;

    void addBefore(ListNode *cur, int val) {
        ListNode *node = new ListNode(val);
        node->prev = cur->prev;
        node->next = cur;
        cur->prev->next = node;
        cur->prev = node;
        size++;
    }

public:
    MyQueue() {
        fake_head = new ListNode();
        fake_tail = new ListNode();
        fake_head->next = fake_tail;
        fake_tail->prev = fake_head;
    }

    void push(int x) {
        addBefore(fake_tail, x);  // 在尾部插入(入队)
    }

    int pop() {
        if (empty()) return -1;
        ListNode *front = fake_head->next;
        int val = front->val;
        // 摘除队首
        front->prev->next = front->next;
        front->next->prev = front->prev;
        delete front;
        size--;
        return val;
    }

    int peek() {
        if (empty()) return -1;
        return fake_head->next->val;  // 队首元素
    }

    bool empty() {
        return size == 0;
    }
};
```

---

## 4. 栈和队列的互相实现(经典面试题)

这是 LeetCode 225 和 232 的考点:用一种结构模拟另一种结构的顺序。核心技巧:**用两个辅助结构倒腾元素**。

### 4.1 用两个队列实现栈(LeetCode 225)
