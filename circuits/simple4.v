module top (
    a , b , c );
    input a , b , c;
    output out;
    wire n1, n2, n3, n4, n5, n6, n7, n8;
    assign n1 = a & b;
    assign n2 = a & c;
    assign n3 = b & c;
    assign n4 = n1 & n2;
    assign n5 = n1 & n3;
    assign n6 = n2 & n3;
    assign n7 = n4 & n5;
    assign n8 = n5 & n6;
    assign out = n7 & n8;
endmodule