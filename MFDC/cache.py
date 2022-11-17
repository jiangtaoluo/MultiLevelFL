"""
 存储管理
"""
import random

random.seed(2019)
stream = []  # 页面走向
Len = 320  #


# def produce_addstream():
#     """
#     1）	50%的指令是顺序执行的
#     2）	25%的指令是均匀分布在前地址部分
#     3）	25%的指令是均匀分布在后地址部分
#     :return:
#     """
#     ret = []
#     count = 320
#     while count > 0:
#         count -= 4
#         m = random.randint(0, 319)
#         pre = random.randint(0, m + 1)
#         next = random.randint(pre + 2, 319)
#         ret.extend([m + 1, pre, pre + 1, next])
#         print(m + 1, pre, pre + 1, next)
#     print("地址流", ret)
#     return [(i - 1) // 10 for i in ret]

# stream = produce_addstream()

def fifo(seq, size):
    ps = [] * size
    miss = 0
    for page in seq:
        if page not in ps:
            if len(ps) < size:
                ps.append(page)
            else:
                ps.pop(0)
                ps.append(page)
            miss += 1
    print("内存为{}k时的命中率为{:.2%}".format(size, 1 - (miss / len(seq))))
    # miss = 0


def lru(seq):
    ps = []
    miss = 0
    for size in range(4, 33):
        for page in seq:
            if page not in ps:
                if len(ps) < size:
                    ps.append(page)
                else:
                    ps.pop(0)
                    ps.append(page)
                miss += 1
            else:
                ps.append(ps.pop(ps.index(page)))  # 弹出后插入到最近刚刚访问的一端
        print("内存为{}k时的命中率为{:.2%}".format(size, 1 - miss / Len))
        miss = 0


def opt(seq):
    """"
    最佳置换算法，其所选择的被淘汰的页面将是以后永不使用的，或是在最长（未来）时间内不再被访问的页面。
    """

    def find_eliminated(start):
        temp = {}
        for _, i in enumerate(seq[start:]):
            if i in ps and i not in temp:
                temp[ps.index(i)] = _
            if len(temp) == len(ps):
                break
        if len(temp) < len(ps):
            for i in range(len(ps)):
                if i not in temp:
                    return i
        # all in ps, find max one
        mx, j = 0, 0
        for i, v in temp.items():
            if v > mx:
                mx = v
                j = i
        return j

    ps, miss, eliminated = [], 0, -1
    for size in range(4, 33):
        for index, page in enumerate(seq):
            if page not in ps:
                if len(ps) < size:
                    ps.append(page)
                else:
                    ps.pop(find_eliminated(index + 1))
                    ps.append(page)
                miss += 1
        print("内存为{}k时的命中率为{:.2%}".format(size, 1 - miss / Len))
        miss = 0


def lfu(seq):
    """
    LFU（Least Frequently Used）最近最少使用算法。它是基于“如果一个数据在最近一段时间内使用次数很少，
    那么在将来一段时间内被使用的可能性也很小”的思路。
    """

    ps, miss, bad, bad_i = {}, 0, 1 << 31 - 1, 0
    for size in range(4, 33):
        for i, page in enumerate(seq):
            if page not in ps:
                if len(ps) < size:  # 内存还未满
                    ps[page] = 1
                else:
                    for j, v in ps.items():
                        if v < bad:
                            bad, bad_i = v, j
                    ps.pop(bad_i)
                    ps[page] = 1
                    bad, bad_i = 2 ** 32 - 1, 0
                miss += 1
            else:
                ps[page] += 1
        print("内存为{}k时的命中率为{:.2%}".format(size, 1 - miss / Len))
        miss = 0


pair = {1: opt, 2: fifo, 3: lru, 4: lfu}
if __name__ == "__main__":
    prompt = """"There are algorithms in the program
            1、	Optimization algorithm
            2、	First in first out algorithm
            3、	Least recently used algorithm
            4、	Least frequently used algorithm
            """
    print("Start memory management")
    print("Producing address flow, wait for a while, please...")
    stream = produce_addstream()
    print("地址页号流", stream)
    while True:
        print(prompt)
        opt = int(input("Select an algorithm number, please."))
        while opt not in pair:
            print("There is not the algorithm in the program!")
            print(prompt)
            opt = int(input("Select an algorithm number, please."))
        pair[opt](stream)  # 执行对应的算法
        if input("Do you want try again with another algorithm(y/n)") == "n":
            break

