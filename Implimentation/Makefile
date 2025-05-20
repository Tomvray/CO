# ------------  project settings ------------
CC      := gcc
CFLAGS  := -std=c11 -Wall -Wextra -O2
LDLIBS  := -lm                            # e.g.  -lm  if you need libm
TARGET  := app                         # final executable name
SRCS    := main.c algorithms.c utils.c
OBJS    := $(SRCS:.c=.o)

# ------------  explicit dependencies ------------
# utils.c needs parmeters.h
utils.o      : utils.c utils.h parameters.h

# algorithms.c needs utils.{h,c} (+ parmeters.h indirectly)
algorithms.o : algorithms.c algorithms.h utils.h parameters.h

# main.c only uses algorithms (which already pulls the rest)
main.o       : main.c algorithms.h

# ------------  build rules ------------
.PHONY : all clean
all : $(TARGET)

$(TARGET) : $(OBJS)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@

# Generic “compile-a-.c-to-.o” pattern
%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean :
	$(RM) $(OBJS) $(TARGET)

re : clean all
