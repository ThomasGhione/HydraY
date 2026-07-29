SHELL := /bin/bash
ROOT_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
OUTPUT_DIR := $(ROOT_DIR)/output
TESTS_DIR := $(ROOT_DIR)/tests
SCRIPT_DIR := $(ROOT_DIR)/script
DOC_DIR := $(ROOT_DIR)/doc
ANALYSIS_DIR := $(DOC_DIR)/analysis-output
SCAN_BUILD_DIR := $(ROOT_DIR)/scan-build-report
ANALYSIS_LOG := $(ROOT_DIR)/analysis.log
COMPLEXITY_REPORT := $(ROOT_DIR)/complexity-report.csv
VALGRIND_REPORT := $(ROOT_DIR)/valgrind-report.txt
GPROF_OUTPUT := $(ROOT_DIR)/gmon.out
PROFILE_IMAGE := $(ROOT_DIR)/output.png

NUMBER_OF_CORES := $(shell nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
ifeq ($(MAKELEVEL),0)
ifeq ($(filter -j%,$(MAKEFLAGS)),)
MAKEFLAGS += -j$(NUMBER_OF_CORES)
endif
endif

# Compiler variables
CXX = g++
TEST_FLAGS = -std=c++23 -Wall -Wextra -Wpedantic -O3 -DDEBUG -fopenmp -march=native -flto=8 -fext-numeric-literals -g

# -O3: Aggressive inlining + vectorization (better than -O2 and safer than -Ofast)
# -march=native: Use all CPU features (SSE4.2, AVX2, BMI2, POPCNT)
# -mtune=native: Optimize instruction scheduling for your CPU
# -flto=auto: Let compiler decide LTO threads (usually better than fixed number)
# -fno-math-errno: Don't set errno on math functions (safe for chess)
# -fno-trapping-math: Assume floating point ops don't trap (safe, we don't use FP exceptions)
# -funroll-loops: Unroll hot loops (good for bitboard operations)
# Balanced production flags (safety + speed)
PRODFLAGS = -std=c++23 -Wall -Wextra -Wpedantic -fopenmp -march=native -mtune=native \
		-flto=auto -fno-math-errno -fno-trapping-math -funroll-loops \
		-ffunction-sections -fdata-sections -O3 

# Uncomment if codebase is exception-free

# Windows cross-compiler (requires mingw-w64)
WIN_CXX = x86_64-w64-mingw32-g++
WIN_CC = x86_64-w64-mingw32-gcc
WIN_PRODFLAGS = -std=c++23 -Wall -Wextra -Wpedantic -O3 -DDEBUG -march=x86-64-v3 -static -static-libgcc -static-libstdc++ -fopenmp -flto=4 -fext-numeric-literals

# Static-analysis tools: cppcheck and lizard come from the system now
# (install: apt/brew cppcheck, `pip install lizard`). gprof2dot stays vendored.
CPPCHECK = cppcheck
COMPILATION_DB = $(ROOT_DIR)/compile_commands.json
LIZARD = lizard
GPROF2DOT = $(SCRIPT_DIR)/gprof2dot.py
LLM_ANALYSIS_PROMPT = $(DOC_DIR)/static-analysis-summary-prompt.txt

# Output binaries
NAME_APP = $(ROOT_DIR)/chess
TEST_APP = $(TESTS_DIR)/test
PERF_APP = $(TESTS_DIR)/perf
TT_HP_BENCH_APP = $(TESTS_DIR)/tt_hugepage_bench
NAME_APP_WIN = $(ROOT_DIR)/chess.exe
GENERATED_BINS = $(NAME_APP) $(NAME_APP_WIN) $(TEST_APP) $(PERF_APP) $(TT_HP_BENCH_APP)

# File paths by module
MAIN_SRC = $(ROOT_DIR)/main.cpp
ENGINE_SRCS = $(wildcard $(ROOT_DIR)/engine/*.cpp) $(wildcard $(ROOT_DIR)/engine/eval/*.cpp) $(wildcard $(ROOT_DIR)/engine/eval/*/*.cpp) $(wildcard $(ROOT_DIR)/engine/search/*.cpp) $(wildcard $(ROOT_DIR)/engine/sort/*.cpp) $(wildcard $(ROOT_DIR)/engine/time/*.cpp) $(wildcard $(ROOT_DIR)/engine/movegen/*.cpp) $(wildcard $(ROOT_DIR)/engine/opening/*.cpp) $(wildcard $(ROOT_DIR)/engine/syzygy/*.cpp)
SYZYGY_C_SRC = $(ROOT_DIR)/engine/syzygy/tbprobe.c
SYZYGY_C_OBJ = $(OUTPUT_DIR)/engine/syzygy/tbprobe.o
DRIVER_SRCS = $(wildcard $(ROOT_DIR)/driver/*.cpp)
BOARD_SRCS = $(wildcard $(ROOT_DIR)/board/*.cpp)
UCI_SRCS = $(wildcard $(ROOT_DIR)/uci/*.cpp)
NNUE_SRCS = $(wildcard $(ROOT_DIR)/nnue/*.cpp)

# Module source paths in a single variable
ALL_MODULE_SRCS = $(ENGINE_SRCS) $(DRIVER_SRCS) $(BOARD_SRCS) $(UCI_SRCS) $(NNUE_SRCS)

# Header file paths (excluding stockfish and test utils)
ENGINE_HDRS = $(wildcard $(ROOT_DIR)/engine/*.hpp) $(wildcard $(ROOT_DIR)/engine/eval/*.hpp) $(wildcard $(ROOT_DIR)/engine/search/*.hpp) $(wildcard $(ROOT_DIR)/engine/sort/*.hpp) $(wildcard $(ROOT_DIR)/engine/time/*.hpp) $(wildcard $(ROOT_DIR)/engine/movegen/*.hpp) $(wildcard $(ROOT_DIR)/engine/opening/*.hpp)
DRIVER_HDRS = $(wildcard $(ROOT_DIR)/driver/*.hpp)
BOARD_HDRS = $(wildcard $(ROOT_DIR)/board/*.hpp)
UCI_HDRS = $(wildcard $(ROOT_DIR)/uci/*.hpp)
NNUE_HDRS = $(wildcard $(ROOT_DIR)/nnue/*.hpp)

# All header paths
ALL_MODULE_HDRS = $(ENGINE_HDRS) $(DRIVER_HDRS) $(BOARD_HDRS) $(UCI_HDRS) $(NNUE_HDRS)

# All files to analyze (.cpp and .hpp)
ALL_ANALYSIS_FILES = $(MAIN_SRC) $(ENGINE_SRCS) $(DRIVER_SRCS) $(BOARD_SRCS) \
			$(ENGINE_HDRS) $(DRIVER_HDRS) $(BOARD_HDRS)


# Helper to generate .o paths inside output/
MAIN_OBJ = $(patsubst $(ROOT_DIR)/%.cpp,$(OUTPUT_DIR)/%.o,$(MAIN_SRC))
MODULE_OBJS = $(patsubst $(ROOT_DIR)/%.cpp,$(OUTPUT_DIR)/%.o,$(ALL_MODULE_SRCS))
ALL_OBJS = $(MAIN_OBJ) $(MODULE_OBJS) $(SYZYGY_C_OBJ)

# Test file paths
TEST_MAIN_SRC = $(TESTS_DIR)/mainTest.cpp
TEST_ENGINE_SRCS = $(wildcard $(ROOT_DIR)/engine/test/*.cpp)
TEST_DRIVER_SRCS = $(wildcard $(ROOT_DIR)/driver/test/*.cpp)
TEST_BOARD_SRCS = $(wildcard $(ROOT_DIR)/board/test/*.cpp)

# Test file paths in a single variable
ALL_TEST_MODULE_SRCS = $(TEST_ENGINE_SRCS) $(TEST_DRIVER_SRCS) $(TEST_BOARD_SRCS)

# Test .o file paths in output/
TEST_MAIN_OBJ = $(patsubst $(ROOT_DIR)/%.cpp,$(OUTPUT_DIR)/%.o,$(TEST_MAIN_SRC))
TEST_OBJS = $(patsubst $(ROOT_DIR)/%.cpp,$(OUTPUT_DIR)/%.o,$(ALL_TEST_MODULE_SRCS))

# Performance-test file paths
PERF_MAIN_SRC = $(TESTS_DIR)/mainPerformanceTest.cpp
PERF_ENGINE_SRCS = $(wildcard $(ROOT_DIR)/engine/test/performance-test/*.cpp)

# Performance-test paths in a single variable
ALL_PERF_MODULE_SRCS = $(PERF_ENGINE_SRCS)

# Performance-test .o paths in output/
PERF_MAIN_OBJ = $(patsubst $(ROOT_DIR)/%.cpp,$(OUTPUT_DIR)/%.o,$(PERF_MAIN_SRC))
PERF_OBJS = $(patsubst $(ROOT_DIR)/%.cpp,$(OUTPUT_DIR)/%.o,$(ALL_PERF_MODULE_SRCS))

# Huge-page TT benchmark
TT_HP_BENCH_SRC = $(TESTS_DIR)/tt_hugepage_bench.cpp
TT_HP_BENCH_OBJ = $(patsubst $(ROOT_DIR)/%.cpp,$(OUTPUT_DIR)/%.o,$(TT_HP_BENCH_SRC))

# Auto-generated dependency files (header dependencies)
DEPFILES = $(ALL_OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(TEST_MAIN_OBJ:.o=.d) $(PERF_OBJS:.o=.d) $(PERF_MAIN_OBJ:.o=.d) $(TT_HP_BENCH_OBJ:.o=.d)

# Main targets
.PHONY: all prod prod_windows parallel_prod prod_sequential debug test perf tt-huge-bench all-tests analyze analyze-setup analyze-cppcheck analyze-clang-tidy analyze-iwyu analyze-scan-build analyze-gcc-analyzer analyze-cppclean analyze-lizard analyze-summary complexity test-valgrind cls cls-compile-files get-image help debug-vars

# Default target for plain `make`
all: PRODFLAGS += -DDEBUG -fomit-frame-pointer
all: prod

chess: $(NAME_APP)

# Parallel production build
# Note: faster after the first build but keeps .o files
prod: $(NAME_APP)
	@printf "\nBuild completed: $(NAME_APP)\n\n"

# Explicit alias for parallel build
parallel_prod: prod

# Windows production build (cross-compile from Linux)
# tbprobe.c compiled separately as C (not C++) to avoid name mangling issues
prod_windows:
	@printf "\nCompiling Windows binary..."
	$(WIN_CC) -O3 -std=c11 -Wall -c $(SYZYGY_C_SRC) -o $(OUTPUT_DIR)/tbprobe_win.o
	$(WIN_CXX) $(WIN_PRODFLAGS) $(MAIN_SRC) $(ALL_MODULE_SRCS) $(OUTPUT_DIR)/tbprobe_win.o -o $(NAME_APP_WIN)
	@printf "\nBuild completed: $(NAME_APP_WIN)\n\n"

# Sequential production build
# Note: no slower, but does not generate .o files
prod_sequential:
	$(CXX) $(PRODFLAGS) $(MAIN_SRC) $(ALL_MODULE_SRCS) -o $(NAME_APP)
	@printf "\nBuild completed: $(NAME_APP)\n\n"

# Build command for debug executable
# Adds -g to include debug symbols
debug: PRODFLAGS += -DDEBUG -g -pg -O1
debug: $(NAME_APP)
	@printf "\nDebug build completed: $(NAME_APP)\n\n"

# Build command for test executable
test: $(TEST_APP)
	@printf "\nTest binary built: $(TEST_APP)\n\n"

# Build command for performance-test executable
perf: $(PERF_APP)
	@printf "\nPerformance-test binary built: $(PERF_APP)\n\n"

# Build command for huge-page TT benchmark
tt-huge-bench: $(TT_HP_BENCH_APP)
	@printf "\nTT huge-page benchmark built: $(TT_HP_BENCH_APP)\n\n"

# Command to run all tests (functional + performance)
all-tests: test perf
	@printf "\n=== Running functional tests ===\n"
	$(TEST_APP)
	@printf "\n=== Running performance tests ===\n"
	$(PERF_APP)
	@printf "\nAll tests completed\n\n"

# Final executable generation: 'chess'
$(NAME_APP): $(ALL_OBJS)
	@printf "\nLinking $(NAME_APP)..."
	$(CXX) $(PRODFLAGS) $(ALL_OBJS) -o $(NAME_APP)

# .o file generation
$(OUTPUT_DIR)/%.o: $(ROOT_DIR)/%.cpp
	@printf "\nCompiling $<..."
	@mkdir -p $(dir $@)
	$(CXX) $(PRODFLAGS) -MMD -MP -c $< -o $@

# embedded.cpp pulls the network blob in via the assembler's .incbin, which
# -MMD cannot track: declare the data dependency explicitly.
$(OUTPUT_DIR)/nnue/embedded.o: $(ROOT_DIR)/nnue/net/hydray.nnue

# tbprobe.c compiled as C (not C++) to avoid name mangling issues
$(SYZYGY_C_OBJ): $(SYZYGY_C_SRC)
	@printf "\nCompiling $<..."
	@mkdir -p $(dir $@)
	gcc -O3 -march=native -std=c11 -Wall -MMD -MP -c $< -o $@

# Final executable generation: 'outputTest'
$(TEST_APP): $(MODULE_OBJS) $(SYZYGY_C_OBJ) $(TEST_OBJS) $(TEST_MAIN_OBJ)
	@printf "\nLinking test $(TEST_APP)..."
	$(CXX) $(TEST_FLAGS) $(MODULE_OBJS) $(SYZYGY_C_OBJ) $(TEST_OBJS) $(TEST_MAIN_OBJ) -o $(TEST_APP)

# Final executable generation: 'outputPerformance'
$(PERF_APP): $(MODULE_OBJS) $(SYZYGY_C_OBJ) $(PERF_OBJS) $(PERF_MAIN_OBJ)
	@printf "\nLinking performance test $(PERF_APP)..."
	$(CXX) $(TEST_FLAGS) $(MODULE_OBJS) $(SYZYGY_C_OBJ) $(PERF_OBJS) $(PERF_MAIN_OBJ) -o $(PERF_APP)

# Final executable generation: TT huge-page benchmark
$(TT_HP_BENCH_APP): $(MODULE_OBJS) $(TT_HP_BENCH_OBJ)
	@printf "\nLinking TT huge-page benchmark $(TT_HP_BENCH_APP)..."
	$(CXX) $(PRODFLAGS) $(MODULE_OBJS) $(TT_HP_BENCH_OBJ) -o $(TT_HP_BENCH_APP)

# Complete static-analysis suite
analyze: analyze-setup
	@$(MAKE) -C $(ROOT_DIR) analyze-cppcheck
	@$(MAKE) -C $(ROOT_DIR) analyze-clang-tidy
	@$(MAKE) -C $(ROOT_DIR) analyze-iwyu
	@$(MAKE) -C $(ROOT_DIR) analyze-scan-build
	@$(MAKE) -C $(ROOT_DIR) analyze-gcc-analyzer
	@$(MAKE) -C $(ROOT_DIR) analyze-cppclean
	@$(MAKE) -C $(ROOT_DIR) analyze-lizard
	@$(MAKE) -C $(ROOT_DIR) analyze-summary

analyze-setup:
	@printf "\n=== Complete Static Analysis Suite ===\n"
	@printf "This will take some minutes...\n"
	@printf "Analyzing %s files (.cpp + .hpp)...\n\n" $(words $(ALL_ANALYSIS_FILES))
	@mkdir -p $(ANALYSIS_DIR)
	@rm -f $(ANALYSIS_LOG)
	@rm -rf $(SCAN_BUILD_DIR)
	@rm -f $(COMPLEXITY_REPORT)
	@printf "=== STATIC ANALYSIS REPORT ===\n" > $(ANALYSIS_LOG)
	@printf "Date: %s\n" "$$(date '+%Y-%m-%d %H:%M:%S')" >> $(ANALYSIS_LOG)
	@printf "Files analyzed: %s\n\n" $(words $(ALL_ANALYSIS_FILES)) >> $(ANALYSIS_LOG)
	@if command -v bear >/dev/null 2>&1; then \
		cd $(ROOT_DIR) && bear -- $(MAKE) prod > /dev/null 2>&1; \
	else \
		printf "bear not found: skipping compile_commands.json generation\n"; \
	fi
	@printf "[0/7] Generating compilation database\n"

analyze-cppcheck:
	@printf "\n========================================\n" >> $(ANALYSIS_LOG)
	@printf "1. CPPCHECK ANALYSIS\n" >> $(ANALYSIS_LOG)
	@printf "========================================\n\n" >> $(ANALYSIS_LOG)
	@if command -v $(CPPCHECK) >/dev/null 2>&1; then \
		$(CPPCHECK) $(ALL_ANALYSIS_FILES) --check-level=exhaustive --enable=all --suppress=missingIncludeSystem --inline-suppr --std=c++23 --quiet >> $(ANALYSIS_LOG) 2>&1; \
		printf "[1/7] Running cppcheck terminated\n"; \
	else \
		printf "cppcheck not found - skipping (install it or 'pip install'/apt)\n" >> $(ANALYSIS_LOG); \
		printf "[1/7] Running cppcheck skipped\n"; \
	fi

analyze-clang-tidy:
	@printf "\n========================================\n" >> $(ANALYSIS_LOG)
	@printf "2. CLANG-TIDY ANALYSIS\n" >> $(ANALYSIS_LOG)
	@printf "========================================\n\n" >> $(ANALYSIS_LOG)
	@clang-tidy -p $(COMPILATION_DB) $(MAIN_SRC) $(ALL_MODULE_SRCS) $(ALL_MODULE_HDRS) >> $(ANALYSIS_LOG) 2>&1 || echo "clang-tidy exited with error code $$?" >> $(ANALYSIS_LOG)
	@printf "[2/7] Running clang-tidy terminated\n"

analyze-iwyu:
	@printf "\n========================================\n" >> $(ANALYSIS_LOG)
	@printf "3. INCLUDE-WHAT-YOU-USE ANALYSIS\n" >> $(ANALYSIS_LOG)
	@printf "========================================\n\n" >> $(ANALYSIS_LOG)
	@for file in $(MAIN_SRC) $(ALL_MODULE_SRCS); do \
		printf "Checking $$file...\n" >> $(ANALYSIS_LOG); \
		include-what-you-use -std=c++23 -fopenmp -DDEBUG -march=native $$file >> $(ANALYSIS_LOG) 2>&1 || true; \
	done 2>/dev/null
	@printf "[3/7] Running include-what-you-use terminated\n"

analyze-scan-build:
	@printf "\n========================================\n" >> $(ANALYSIS_LOG)
	@printf "4. SCAN-BUILD (CLANG STATIC ANALYZER)\n" >> $(ANALYSIS_LOG)
	@printf "========================================\n\n" >> $(ANALYSIS_LOG)
	@printf "Path-sensitive dataflow analysis in progress...\n" >> $(ANALYSIS_LOG)
	@$(MAKE) -C $(ROOT_DIR) cls-compile-files > /dev/null
	@scan-build -o $(SCAN_BUILD_DIR) --status-bugs -v -analyzer-config aggressive-binary-operation-simplification=true $(MAKE) -C $(ROOT_DIR) prod >> $(ANALYSIS_LOG) 2>&1 || echo "scan-build exited with error code $$?" >>  $(ANALYSIS_LOG)
	@printf "\n--- Scan-build text summary ---\n" >> $(ANALYSIS_LOG)
	@if [ -d $(SCAN_BUILD_DIR) ] && [ -n "$$(ls -A $(SCAN_BUILD_DIR) 2>/dev/null)" ]; then \
		printf "HTML Report location: $(SCAN_BUILD_DIR)/\n" >> $(ANALYSIS_LOG); \
		find $(SCAN_BUILD_DIR) -name "*.html" -type f | head -1 | xargs -I {} printf "Main report: {}\n" >> $(ANALYSIS_LOG) 2>&1; \
		printf "\nBugs found by scan-build:\n" >> $(ANALYSIS_LOG); \
		if ! find $(SCAN_BUILD_DIR) -name "*.html" -exec grep -h "<!-- BUGTYPE" {} \; 2>/dev/null | sort | uniq -c >> $(ANALYSIS_LOG); then \
			printf "No bugs found or unable to parse HTML reports\n" >> $(ANALYSIS_LOG); \
		fi; \
	else \
		printf "No bugs detected by scan-build!\n" >> $(ANALYSIS_LOG); \
	fi
	@printf "[4/7] Running scan-build terminated\n"

analyze-gcc-analyzer:
	@printf "\n========================================\n" >> $(ANALYSIS_LOG)
	@printf "5. GCC STATIC ANALYZER\n" >> $(ANALYSIS_LOG)
	@printf "========================================\n\n" >> $(ANALYSIS_LOG)
	@$(MAKE) -C $(ROOT_DIR) cls-compile-files > /dev/null
	@$(CXX) -fanalyzer -fsyntax-only $(PRODFLAGS) $(MAIN_SRC) $(ALL_MODULE_SRCS) >> $(ANALYSIS_LOG) 2>&1 || echo "GCC analyzer exited with error code $$?" >> $(ANALYSIS_LOG)
	@printf "[5/7] Running GCC analyzer terminated\n"

analyze-cppclean:
	@printf "\n========================================\n" >> $(ANALYSIS_LOG)
	@printf "6. CPPCLEAN (DEAD CODE & DEPENDENCIES)\n" >> $(ANALYSIS_LOG)
	@printf "========================================\n\n" >> $(ANALYSIS_LOG)
	@cd $(ROOT_DIR) && cppclean . >> $(ANALYSIS_LOG) 2>&1 || echo "cppclean exited with error code $$?" >> $(ANALYSIS_LOG)
	@printf "[6/7] Running cppclean terminated\n"

analyze-lizard:
	@printf "\n========================================\n" >> $(ANALYSIS_LOG)
	@printf "7. LIZARD COMPLEXITY ANALYSIS\n" >> $(ANALYSIS_LOG)
	@printf "========================================\n\n" >> $(ANALYSIS_LOG)
	@if command -v $(LIZARD) >/dev/null 2>&1; then \
		$(LIZARD) -l cpp -w -L 60 -C 15 --csv $(ROOT_DIR) > $(COMPLEXITY_REPORT) 2>&1 || true; \
		$(LIZARD) -l cpp -w -L 60 -C 15 $(ROOT_DIR) >> $(ANALYSIS_LOG) 2>&1 || true; \
		printf "[7/7] Running lizard terminated\n"; \
	else \
		printf "Lizard not found - skipping ('pip install lizard')\n" >> $(ANALYSIS_LOG); \
		printf "[7/7] Running lizard skipped\n"; \
	fi

analyze-summary:
	@printf "\n========================================\n" >> $(ANALYSIS_LOG)
	@printf "COMPLETE ANALYSIS FINISHED\n" >> $(ANALYSIS_LOG)
	@printf "========================================\n" >> $(ANALYSIS_LOG)
	@printf "\n========================================\n"
	@printf "Complete analysis finished!\n"
	@printf "========================================\n"
	@printf "\nReports generated:\n"
	@printf "   1. $(ANALYSIS_LOG) (main report)\n"
	@printf "   2. $(SCAN_BUILD_DIR)/*/index.html (interactive HTML)\n"
	@if [ -f $(COMPLEXITY_REPORT) ]; then \
		printf "   3. $(COMPLEXITY_REPORT) (spreadsheet)\n"; \
	fi
	@printf "\n[7/7] Generating LLM prompt...\n\n"
	@printf "=== LLM PROMPT (copy below) ===\n\n"
	@if [ -f $(LLM_ANALYSIS_PROMPT) ]; then \
		cat $(LLM_ANALYSIS_PROMPT); \
	else \
		printf "Prompt template not found: $(LLM_ANALYSIS_PROMPT)\n"; \
	fi
	@printf "\n\n=== END PROMPT ===\n"

# Complexity-analysis alias
complexity: analyze-lizard

# Memory leak detection with report saved to file
test-valgrind: $(TEST_APP)
	@mkdir -p $(ANALYSIS_DIR)
	@if ! command -v valgrind >/dev/null 2>&1; then \
		printf "valgrind not found. Install it with: sudo apt install valgrind\n"; \
		exit 1; \
	fi
	@printf "\nRunning valgrind on $(TEST_APP)...\n"
	@valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
		--error-exitcode=1 $(TEST_APP) > $(VALGRIND_REPORT) 2>&1
	@printf "Valgrind report: $(VALGRIND_REPORT)\n\n"

# Include dependency files generated by -MMD -MP
-include $(DEPFILES)

# Clean object files and binaries only
cls-compile-files:
	rm -rf $(OUTPUT_DIR)
	rm -f $(GENERATED_BINS)

# Clean temporary files
cls: cls-compile-files
	@printf "\nCleaning..."
	rm -f $(DOC_DIR)/main-doc.{aux,log,pdf,toc}
	rm -f $(COMPILATION_DB)
	#rm -rf $(ANALYSIS_DIR)
	rm -rf $(GPROF_OUTPUT)
	@printf "\nClean completed\n\n"

# Generate a profiling call graph image from gprof output.
get-image:
	gprof $(NAME_APP) $(GPROF_OUTPUT) | python3 $(GPROF2DOT) -s -w | dot -Tpng -Gdpi=200 -o $(PROFILE_IMAGE)
	@printf "\nImage created: $(PROFILE_IMAGE)\n\n"

# Command help
help:
	@printf "\n=== Chess Build System ===\n"
	@printf "\nBUILD TARGETS:\n"
	@printf "  make prod           - Optimized production build\n"
	@printf "  make prod_windows   - Windows cross-build (requires mingw-w64)\n"
	@printf "  make debug          - Build with debug symbols\n"
	@printf "  make test           - Build functional tests\n"
	@printf "  make perf           - Build performance tests\n"
	@printf "  make all-tests      - Run functional and performance tests\n"
	@printf "\nSTATIC ANALYSIS:\n"
	@printf "  make analyze        - Full static-analysis suite (7 tools, about 5 min)\n"
	@printf "\nRUNTIME TESTING:\n"
	@printf "  make test-valgrind  - Memory leak detection (valgrind)\n"
	@printf "\nUTILITIES:\n"
	@printf "  make cls            - Remove compiled files and generated reports\n"
	@printf "  make debug-vars     - Print makefile variables\n"
	@printf "  make help           - Show this help message\n"
	@printf "\nGENERATED REPORTS:\n"
	@printf "  - $(ANALYSIS_LOG)                    (main text report)\n"
	@printf "  - $(SCAN_BUILD_DIR)/*.html       (interactive HTML)\n"
	@printf "  - $(COMPLEXITY_REPORT)     (complexity metrics)\n"
	@printf "  - $(VALGRIND_REPORT)            (memory analysis)\n"
	@printf "\nTOOL LOCATIONS:\n"
	@printf "  - lizard: $(LIZARD)\n"
	@printf "  - valgrind: install with 'sudo apt install valgrind'\n"
	@printf "\n==================================\n"

# Print variables
.PHONY: debug-vars
debug-vars:
	@printf "\n"
	@printf "SHELL = $(SHELL)"
	@printf "Compiler = $(CXX)"
	@printf "Test flags = $(TEST_FLAGS)"
	@printf "Production flags = $(PRODFLAGS)"
	@printf "Module sources = $(ALL_MODULE_SRCS)"
	@printf "Test module sources = $(ALL_TEST_MODULE_SRCS)"
	@printf "MODULE_OBJS = $(MODULE_OBJS)"
