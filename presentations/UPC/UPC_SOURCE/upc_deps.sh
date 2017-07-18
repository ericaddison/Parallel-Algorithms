echo "setting environment variables"
LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
C_INCLUDE_PATH=/usr/include/x86_64-linux-gnu
CPLUS_INCLUDE_PATH=/usr/include/x86_64-linux-gnu
UPC_FILE=upc-*.tar.gz
echo "Installing dependencies."
sudo apt-get install libgmp3-dev libmpfr-dev libmpc-dev libnuma-dev

if [ ! -f $UPC_FILE ]
then
	echo "Downloading the upc package."
	#wget -O upc-5.2.0.1-x86_64-linux-ubuntu12.4.tar.gz http://www.gccupc.org/gupc-5201-1/30-gupc-5201-x8664-ubuntu-1204/file 
fi

if [ ! -d gupc ]
then
echo "Cloning the UPC repo....."
	svn checkout svn://gcc.gnu.org/svn/gcc/branches/gupc
	# git clone git@github.com:Intrepid/llvm-upc.git
fi

#tar xpf $UPC_FILE

cd build
../gupc/configure  	--disable-multilib \
					--prefix=/usr/local \
					--enable-languages=c,c++ \
					--enable-threads=posix  \
 					--enable-shared


make
#if [ "$?" == "0" ]
#then
#	echo "Installing"
#	# make install
#fi

