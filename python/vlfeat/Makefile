##################################################
# VLFeat Python Wrapping
##################################################

include ./MakefileBase

LIBRARY = _vlfeat.so

##################################################

VLFEAT_DIR = ../../vl

SOURCES_C = \
$(VLFEAT_DIR)/aib.c \
$(VLFEAT_DIR)/generic.c \
$(VLFEAT_DIR)/hikmeans.c \
$(VLFEAT_DIR)/ikmeans.c \
$(VLFEAT_DIR)/imopv.c \
$(VLFEAT_DIR)/mathop.c \
$(VLFEAT_DIR)/mathop_sse2.c \
$(VLFEAT_DIR)/pgm.c \
$(VLFEAT_DIR)/rodrigues.c \
$(VLFEAT_DIR)/stringop.c \
$(VLFEAT_DIR)/getopt_long.c \
$(VLFEAT_DIR)/host.c \
$(VLFEAT_DIR)/imopv_sse2.c \
$(VLFEAT_DIR)/mser.c \
$(VLFEAT_DIR)/random.c \
$(VLFEAT_DIR)/sift.c \
$(VLFEAT_DIR)/dsift.c \

SOURCES_CPP = \
mser/vl_erfill.cpp \
mser/vl_mser.cpp \
sift/vl_sift.cpp \
sift/vl_dsift.cpp \
sift/vl_siftdescriptor.cpp \
imop/vl_imsmooth.cpp \
kmeans/vl_ikmeans.cpp \
kmeans/vl_hikmeanspush.cpp \
kmeans/vl_ikmeanspush.cpp \
py_vlfeat.cpp
# misc/vl_binsum.cpp \
# kmeans/vl_hikmeans.cpp \

CFLAGS += -I$(VLFEAT_DIR)/../
CFLAGS += -I/usr/local/Cellar/python/2.7.3/Frameworks/Python.framework/Versions/2.7/include/python2.7
CFLAGS += -I/usr/local/lib/python2.7/site-packages/numpy/core/include
LFLAGS += -L/usr/local/Cellar/python/2.7.3/Frameworks/Python.framework/Versions/2.7/lib
LFLAGS += -L/usr/local/lib
LFLAGS += -msse -shared -lpython2.7

OBJS_C=$(SOURCES_C:.c=.o)
OBJS_CPP=$(SOURCES_CPP:.cpp=.o)
OBJS =  $(OBJS_C) $(OBJS_CPP) 

.cpp.o:
	$(CXX) $(CFLAGS) -c $< -o $@

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@


.SUFFIXES:
.SUFFIXES:  .c .cpp .o

##################################################

.PHONY: all
all: $(LIBRARY)

.PHONY: clean
clean:
	$(RM) $(OBJS) $(LIBRARY)

##################################################

$(LIBRARY): $(OBJS)
	$(CXX) -o $@ $^ $(LFLAGS) -lboost_python-mt

##################################################
