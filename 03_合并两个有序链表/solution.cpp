/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */

// 解法1：迭代法（推荐）
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode dummy(0);
        ListNode* fake_head = &dummy;

        while(list1 && list2) {
            if(list1->val < list2->val) {
                fake_head->next = list1;
                list1 = list1->next;
            } else {
                fake_head->next = list2;
                list2 = list2->next;
            }
            fake_head = fake_head->next;
        }

        // 连接剩余部分（两种写法都可以）
        fake_head->next = list1 ? list1 : list2;
        // 或者：
        // if(!list1) fake_head->next = list2;
        // if(!list2) fake_head->next = list1;

        return dummy.next;
    }
};

// 解法2：递归法
class Solution2 {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        // 递归终止条件
        if (!list1) return list2;
        if (!list2) return list1;

        // 选择较小的节点，递归处理剩余部分
        if (list1->val <= list2->val) {
            list1->next = mergeTwoLists(list1->next, list2);
            return list1;
        } else {
            list2->next = mergeTwoLists(list1, list2->next);
            return list2;
        }
    }
};
