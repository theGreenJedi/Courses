INCLUDE_REPO=https://github.com/masteringmatplotlib/includes.git
INCLUDE_DIR=include
NAME=arch
NOTEBOOK=notebooks/mmpl-$(NAME).ipynb

-include include/common.mk
-include include/graphviz.mk
-include include/types.mk

# The following target is intended for use by project-creators only. When
# creating a new notebook project, add a copy of this Makefile and run this
# target to get the includes set up:
#
# $ make setup-submodule
setup-submodule:
	git submodule add $(INCLUDE_REPO) $(INCLUDE_DIR)

# The 'setup' target needs to be run before the 'project-deps' target,
# so that the includes are present (done by 'make project-setup').
deps: pygraphviz base-deps
	. $(VENV)/bin/activate && \
	pip3.4 install -r requirements/part2.txt

setup:
	@git submodule init
	@git submodule update
	@make project-setup

.DEFAULT_GOAL :=
default: setup
	make run
