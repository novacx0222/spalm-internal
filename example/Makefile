CC      = g++
CFLAGS  = --std=c++20 -O2 -Wall
BIN_DIR = bin

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/example: example.cpp | $(BIN_DIR)
	$(CC) $(CFLAGS) example.cpp -ldl -o $(BIN_DIR)/example

example: $(BIN_DIR)/example

clean:
	rm -rf $(BIN_DIR)

.PHONY: example clean
