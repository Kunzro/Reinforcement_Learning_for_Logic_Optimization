module top (
    a , b , c , d , out );
    input a , b , c , d;
    output out;
    wire n1 , n2 , n3;
    assign n1 = a & b;
    assign n2 = n1 & c;
    assign n3 = n2 & d;
    assign out = n3;
endmodule