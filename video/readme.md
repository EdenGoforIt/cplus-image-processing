# Algorithm

## Four Adjacent Object Detection

```
1:  declare a vector of sets SET[]
2:  declare integers counter=-1, s1, s2
3:  declare A[i.width][i.height]          {a 2D vector or mat initialised to -1}
4:  for y = 1 to i.height do
5:      for x = 1 to i.width do
6:          if (i(x,y) ≠ 0) then
7:              if (i(x-1,y) ≠ 0 OR i(x,y-1) ≠ 0) then
8:                  s1 = A[x-1][y]
9:                  s2 = A[x][y-1]
10:                 if (s1 ≠ -1) then
11:                     i(x,y) → SET[s1]      {insert point i(x,y) into SET[s1]}
12:                     A[x][y] = s1
13:                 end if
14:                 if (s2 ≠ -1) then
15:                     i(x,y) → SET[s2]
16:                     A[x][y] = s2
17:                 end if
18:                 if ((s1 ≠ s2) AND (s1 ≠ -1) AND (s2 ≠ -1)) then
19:                     SET[s1] ∪ SET[s2]     {Union, keep set SET[s1] and empty the other}
20:                     Reset all points of A(x,y) belonging to SET[s2] to s1
21:                     empty SET[s2]
22:                 end if
23:             else
24:                 counter = counter + 1
25:                 Create new set SET[counter]
26:                 i(x,y) → SET[counter]
27:                 A[x][y] = counter
28:             end if
29:         end if
30:     end for
31: end for
```


## Code 