function x = wilkinson(a,b,c)
  delta = (a-c)/2;
  x = c - sign(delta)*(b^2)/(abs(delta)+sqrt((delta^2)+(b^2)));
end