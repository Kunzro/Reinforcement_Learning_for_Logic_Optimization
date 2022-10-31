module top (
    a , b , out );
    input a , b;
    output out;
    wire n1 , n2;
    assign n1 = a & b;
    assign n2 = a & ~b;
    assign out = n1 | n2;
endmodule