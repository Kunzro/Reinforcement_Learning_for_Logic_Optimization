module top (
    a , b , c );
    input a , b , c;
    output out1, out2, out3;
    wire n1, n2, n3, n4, n5, n6, n7, n8, m1, p1, p2, p3, p4;
    assign n1 = a & b;
    assign n2 = a & c;
    assign n3 = b & c;
    assign n4 = n1 & n2;
    assign n5 = n1 & n3;
    assign n6 = n2 & n3;
    assign n7 = n4 & n5;
    assign n8 = n5 & n6;
    assign out1 = n7 & n8;
    assign m1 = n1 & n2;
    assign out2 = m1 & ~n3;
    assign p1 = n3 & n2;
    assign p2 = p1 & n2;
    assign p3 = ~a & p2;
    assign p4 = ~p3 & n6;
    assign out3 = p4 & p3;
endmodule