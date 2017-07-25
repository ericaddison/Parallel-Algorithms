bit x, y; /* signal entering/leaving the section */
byte mutex; /* # of procs in the critical section. */

active proctype A() {
x = 1;
y == 0;
mutex++;
mutex--;
x = 0;
}

active proctype B() {
y = 1;
x == 0;
mutex++;
mutex--;
y = 0;
}

active proctype monitor() {
assert(mutex != 2);
}
