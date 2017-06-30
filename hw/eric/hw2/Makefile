CODE_DIRS = common p1 p2 p3
CLEAN_DIRS=$(addsuffix clean,$(CODE_DIRS))

.PHONY: all $(CODE_DIRS)

all: $(CODE_DIRS)
	@echo ""

$(CODE_DIRS):
	@echo ""
	$(MAKE) -C $@
	
clean: $(CLEAN_DIRS)

%clean:
	@echo ""
	$(MAKE) -C $* clean
