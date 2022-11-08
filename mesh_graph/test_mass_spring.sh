make clean
/usr/bin/clang++  -std=c++11 -W -Wall -Wextra  -I. -isystem ../googletest/googletest/include -L/usr/local/lib/  mass_spring.cpp  -lsfml-graphics -lsfml-window -lsfml-system -framework OpenGL -o mass_spring
./mass_spring data/grid1.nodes data/grid1.tets
