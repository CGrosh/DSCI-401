# Some Basic Illustrations of Python concepts
from functools import reduce
# Illustrates optional/default args
def print_all(a,b,c=1,d=2):
    print(a)
    print(b)
    print(c)
    print(d)

# Illustrates optional/unbounded args
def print_optional(a,b,*c):
    print(a)
    print(b)
    print(c)

# Adding the cache lessens the computational load by checking back in the dictionary
# to see if it has already run this index and returning it as opposed to resolving a problem
# that has already been run
def fib(a,b,n, cache = {}):
    if n in cache:
        return cache[n]
    if n == 1:
        return a
    if  n == 2:
        return b
    else:
        cache[n] = fib(a,b,n-1, cache) + fib(a,b,n-2, cache)
        return cache[n]

def CP(*S):
    if len(S) == 1:
        return [(i) for i in S[0]]
    else:
        return [[i]+[x] for i in S[0] for x in CP(*S[1:])]

def cartesian_prod(*lists):
    if len(lists) == 1:
        return [[x] for x in lists[0]]
    else:
        rest = cartesian_prod(*lists[1:])
        return reduce(lambda x,y: x+y, [[[i] + j for j in rest] for i in lists[0]]) # functional way
        # Manual loop based way
        #cp = []
        #for i in lists[0]:
            #for j in rest:
                #cp.append([i] + j)
        #return cp

# Generate all unique combinations of elements, taken k at a time
def gen_combs(elems, k):
    if k == len(elems):
        return [elems]
    elif k == 1:
        return [[x] for x in elems]
    else:
        return [[elems[0]] + c for c in gen_combs(elems[1:], k-1)] + gen_combs(elems[1:], k)
