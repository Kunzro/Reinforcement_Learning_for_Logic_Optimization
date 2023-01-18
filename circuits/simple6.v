module top (
    a , b , out1, out2 );
    input a , b;
    output out1, out2;
    wire n1 , n2;
    assign n1 = a & b;
    assign n2 = a & ~b;
    assign out1 = n1 | n2;
    assign out2 = 1;
endmodule