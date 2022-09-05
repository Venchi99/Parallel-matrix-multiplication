CC 	= gcc
CFLAGS	= -Wall -O3 -fopenmp
LDFLAGS	= -lm

all: task_11 task_12

task_11:	PartA/task_11.c
	$(CC) $(CFLAGS) -o $@ $? $(LDFLAGS)

task_12:	PartA/task_12.c
	$(CC) $(CFLAGS) -o $@ $? $(LDFLAGS)

.SUFFIXES:	.o .c

%.o : %.c
	$(CC) $(CFLAGS) -c $<
%.s : %.c
	$(CC) $(CFLAGS) -S $<

clean:
	/bin/rm -f *.o *~ task_11 task_12
