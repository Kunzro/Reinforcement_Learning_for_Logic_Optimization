module top (
    a , b , c , out );
    input a , b , c;
    output out;
    wire n1 , n2;
    assign n1 = a & b;
    assign n2 = n1 & c;
    assign out = n2;
endmodule