module top (
    a , b , c , d , e , f , g , h , out );
    input a , b , c , d , e , f , g , h;
    output out;
    wire n1 , n2 , n3 , n4 , n5 , n6 , n7;
    assign n1 = a & b;
    assign n2 = n1 & c;
    assign n3 = n2 & d;
    assign n4 = n3 & e;
    assign n5 = n4 & f;
    assign n6 = n5 & g;
    assign n7 = n6 & h;
    assign out = n7 & a;
endmodule