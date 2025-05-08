.DEFAULT_GOAL := compile

MPICXX = mpic++
FLAGS = -std=c++17 -O3
SRCS = check.cpp template.cpp
EXEC = check

compile: $(EXEC)

$(EXEC): $(SRCS)
	@$(MPICXX) $(FLAGS) -o $(EXEC) $(SRCS)

clean:
	@rm -f $(EXEC)
