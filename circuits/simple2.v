module top (
    a , b , c , out );
    input a , b , c;
    output out;
    wire n1 , n2;
    assign n1 = a & b;
    assign n2 = a & c;
    assign out = n1 & n2;
endmodule