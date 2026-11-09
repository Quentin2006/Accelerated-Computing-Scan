CC     = nvcc
TARGET = scan
SRC    = main.cu

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) a.out
