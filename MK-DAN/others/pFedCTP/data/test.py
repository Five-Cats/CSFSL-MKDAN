s = 72
t0 = 2400
a = 60 * 24 / (t0 / 60)
st = 1 / a
print(a)
print(st)

for i in range(0, s):
    print(i)
    s_t = st * (i % a)
    print(s_t)