OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
cx q[0],q[2];
rz(0.4409554727484849) q[2];
cx q[0],q[2];
cx q[0],q[10];
rz(0.489565418260018) q[10];
cx q[0],q[10];
cx q[14],q[0];
rz(0.5830584638255298) q[0];
cx q[14],q[0];
cx q[1],q[7];
rz(0.5631586122206331) q[7];
cx q[1],q[7];
cx q[1],q[12];
rz(0.44398965526594986) q[12];
cx q[1],q[12];
cx q[17],q[1];
rz(0.4862727032491921) q[1];
cx q[17],q[1];
cx q[2],q[5];
rz(0.384516012955966) q[5];
cx q[2],q[5];
cx q[9],q[2];
rz(0.46258306775972563) q[2];
cx q[9],q[2];
cx q[3],q[5];
rz(0.3988408422003164) q[5];
cx q[3],q[5];
cx q[3],q[13];
rz(0.42428307888555716) q[13];
cx q[3],q[13];
cx q[14],q[3];
rz(0.34637605711543074) q[3];
cx q[14],q[3];
cx q[4],q[8];
rz(0.4892306202296524) q[8];
cx q[4],q[8];
cx q[4],q[10];
rz(0.4928030267680288) q[10];
cx q[4],q[10];
cx q[11],q[4];
rz(0.5150728325336892) q[4];
cx q[11],q[4];
cx q[13],q[5];
rz(0.4233542149231717) q[5];
cx q[13],q[5];
cx q[6],q[9];
rz(0.464419221503683) q[9];
cx q[6],q[9];
cx q[6],q[15];
rz(0.46144853712793393) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(0.48237674115862644) q[6];
cx q[16],q[6];
cx q[7],q[14];
rz(0.49034493561664694) q[14];
cx q[7],q[14];
cx q[15],q[7];
rz(0.4532735394259811) q[7];
cx q[15],q[7];
cx q[8],q[11];
rz(0.5500171314869042) q[11];
cx q[8],q[11];
cx q[16],q[8];
rz(0.5147642213794071) q[8];
cx q[16],q[8];
cx q[15],q[9];
rz(0.5092417550118562) q[9];
cx q[15],q[9];
cx q[13],q[10];
rz(0.44163933037179637) q[10];
cx q[13],q[10];
cx q[17],q[11];
rz(0.515238451514385) q[11];
cx q[17],q[11];
cx q[12],q[16];
rz(0.563069861623951) q[16];
cx q[12],q[16];
cx q[17],q[12];
rz(0.5213705476011179) q[12];
cx q[17],q[12];
rx(5.259618695927884) q[0];
rx(5.259618695927884) q[1];
rx(5.259618695927884) q[2];
rx(5.259618695927884) q[3];
rx(5.259618695927884) q[4];
rx(5.259618695927884) q[5];
rx(5.259618695927884) q[6];
rx(5.259618695927884) q[7];
rx(5.259618695927884) q[8];
rx(5.259618695927884) q[9];
rx(5.259618695927884) q[10];
rx(5.259618695927884) q[11];
rx(5.259618695927884) q[12];
rx(5.259618695927884) q[13];
rx(5.259618695927884) q[14];
rx(5.259618695927884) q[15];
rx(5.259618695927884) q[16];
rx(5.259618695927884) q[17];
cx q[0],q[2];
rz(2.1715207728978663) q[2];
cx q[0],q[2];
cx q[0],q[10];
rz(2.4109043682295797) q[10];
cx q[0],q[10];
cx q[14],q[0];
rz(2.871318407999978) q[0];
cx q[14],q[0];
cx q[1],q[7];
rz(2.7733199845576495) q[7];
cx q[1],q[7];
cx q[1],q[12];
rz(2.1864628493038367) q[12];
cx q[1],q[12];
cx q[17],q[1];
rz(2.394689127718619) q[1];
cx q[17],q[1];
cx q[2],q[5];
rz(1.893580103318072) q[5];
cx q[2],q[5];
cx q[9],q[2];
rz(2.2780276080256834) q[2];
cx q[9],q[2];
cx q[3],q[5];
rz(1.9641238797189708) q[5];
cx q[3],q[5];
cx q[3],q[13];
rz(2.089416225285339) q[13];
cx q[3],q[13];
cx q[14],q[3];
rz(1.7057568161528167) q[3];
cx q[14],q[3];
cx q[4],q[8];
rz(2.4092556283395115) q[8];
cx q[4],q[8];
cx q[4],q[10];
rz(2.4268482323250513) q[10];
cx q[4],q[10];
cx q[11],q[4];
rz(2.536517686084424) q[4];
cx q[11],q[4];
cx q[13],q[5];
rz(2.0848419598227883) q[5];
cx q[13],q[5];
cx q[6],q[9];
rz(2.2870698951581803) q[9];
cx q[6],q[9];
cx q[6],q[15];
rz(2.2724405204699525) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(2.3755031483330957) q[6];
cx q[16],q[6];
cx q[7],q[14];
rz(2.4147431642926) q[14];
cx q[7],q[14];
cx q[15],q[7];
rz(2.23218208526439) q[7];
cx q[15],q[7];
cx q[8],q[11];
rz(2.708603703292202) q[11];
cx q[8],q[11];
cx q[16],q[8];
rz(2.5349979055766676) q[8];
cx q[16],q[8];
cx q[15],q[9];
rz(2.5078020747595122) q[9];
cx q[15],q[9];
cx q[13],q[10];
rz(2.174888484892616) q[10];
cx q[13],q[10];
cx q[17],q[11];
rz(2.53733329010613) q[11];
cx q[17],q[11];
cx q[12],q[16];
rz(2.772882925089715) q[16];
cx q[12],q[16];
cx q[17],q[12];
rz(2.567531291620314) q[12];
cx q[17],q[12];
rx(5.572782596954657) q[0];
rx(5.572782596954657) q[1];
rx(5.572782596954657) q[2];
rx(5.572782596954657) q[3];
rx(5.572782596954657) q[4];
rx(5.572782596954657) q[5];
rx(5.572782596954657) q[6];
rx(5.572782596954657) q[7];
rx(5.572782596954657) q[8];
rx(5.572782596954657) q[9];
rx(5.572782596954657) q[10];
rx(5.572782596954657) q[11];
rx(5.572782596954657) q[12];
rx(5.572782596954657) q[13];
rx(5.572782596954657) q[14];
rx(5.572782596954657) q[15];
rx(5.572782596954657) q[16];
rx(5.572782596954657) q[17];
cx q[0],q[2];
rz(0.9706854729936636) q[2];
cx q[0],q[2];
cx q[0],q[10];
rz(1.0776916694627852) q[10];
cx q[0],q[10];
cx q[14],q[0];
rz(1.2835000713649447) q[0];
cx q[14],q[0];
cx q[1],q[7];
rz(1.2396940681256545) q[7];
cx q[1],q[7];
cx q[1],q[12];
rz(0.9773646890917355) q[12];
cx q[1],q[12];
cx q[17],q[1];
rz(1.0704433398121858) q[1];
cx q[17],q[1];
cx q[2],q[5];
rz(0.8464439857914927) q[5];
cx q[2],q[5];
cx q[9],q[2];
rz(1.018294797722866) q[2];
cx q[9],q[2];
cx q[3],q[5];
rz(0.8779775634653023) q[5];
cx q[3],q[5];
cx q[3],q[13];
rz(0.9339841470708904) q[13];
cx q[3],q[13];
cx q[14],q[3];
rz(0.7624856195549449) q[3];
cx q[14],q[3];
cx q[4],q[8];
rz(1.076954670821092) q[8];
cx q[4],q[8];
cx q[4],q[10];
rz(1.0848186918951823) q[10];
cx q[4],q[10];
cx q[11],q[4];
rz(1.1338417300000911) q[4];
cx q[11],q[4];
cx q[13],q[5];
rz(0.9319394173637049) q[5];
cx q[13],q[5];
cx q[6],q[9];
rz(1.0223367653944158) q[9];
cx q[6],q[9];
cx q[6],q[15];
rz(1.0157973292231959) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(1.0618670684234346) q[6];
cx q[16],q[6];
cx q[7],q[14];
rz(1.0794076390351999) q[14];
cx q[7],q[14];
cx q[15],q[7];
rz(0.9978015178511747) q[7];
cx q[15],q[7];
cx q[8],q[11];
rz(1.2107654228764033) q[11];
cx q[8],q[11];
cx q[16],q[8];
rz(1.1331623771339203) q[8];
cx q[16],q[8];
cx q[15],q[9];
rz(1.1210056442904313) q[9];
cx q[15],q[9];
cx q[13],q[10];
rz(0.9721908645843539) q[10];
cx q[13],q[10];
cx q[17],q[11];
rz(1.1342063108898834) q[11];
cx q[17],q[11];
cx q[12],q[16];
rz(1.2394986993860813) q[16];
cx q[12],q[16];
cx q[17],q[12];
rz(1.1477050357232366) q[12];
cx q[17],q[12];
rx(1.9690159201810378) q[0];
rx(1.9690159201810378) q[1];
rx(1.9690159201810378) q[2];
rx(1.9690159201810378) q[3];
rx(1.9690159201810378) q[4];
rx(1.9690159201810378) q[5];
rx(1.9690159201810378) q[6];
rx(1.9690159201810378) q[7];
rx(1.9690159201810378) q[8];
rx(1.9690159201810378) q[9];
rx(1.9690159201810378) q[10];
rx(1.9690159201810378) q[11];
rx(1.9690159201810378) q[12];
rx(1.9690159201810378) q[13];
rx(1.9690159201810378) q[14];
rx(1.9690159201810378) q[15];
rx(1.9690159201810378) q[16];
rx(1.9690159201810378) q[17];
cx q[0],q[2];
rz(0.9416191405663477) q[2];
cx q[0],q[2];
cx q[0],q[10];
rz(1.0454211295296518) q[10];
cx q[0],q[10];
cx q[14],q[0];
rz(1.2450667777979556) q[0];
cx q[14],q[0];
cx q[1],q[7];
rz(1.202572507234069) q[7];
cx q[1],q[7];
cx q[1],q[12];
rz(0.9480983533462883) q[12];
cx q[1],q[12];
cx q[17],q[1];
rz(1.03838984480764) q[1];
cx q[17],q[1];
cx q[2],q[5];
rz(0.8210979566640141) q[5];
cx q[2],q[5];
cx q[9],q[2];
rz(0.9878028454653169) q[2];
cx q[9],q[2];
cx q[3],q[5];
rz(0.8516872887744664) q[5];
cx q[3],q[5];
cx q[3],q[13];
rz(0.9060168039346211) q[13];
cx q[3],q[13];
cx q[14],q[3];
rz(0.7396536506982557) q[3];
cx q[14],q[3];
cx q[4],q[8];
rz(1.0447061996714255) q[8];
cx q[4],q[8];
cx q[4],q[10];
rz(1.0523347394726272) q[10];
cx q[4],q[10];
cx q[11],q[4];
rz(1.0998898253295648) q[4];
cx q[11],q[4];
cx q[13],q[5];
rz(0.9040333018805184) q[5];
cx q[13],q[5];
cx q[6],q[9];
rz(0.9917237799296431) q[9];
cx q[6],q[9];
cx q[6],q[15];
rz(0.9853801614881907) q[15];
cx q[6],q[15];
cx q[16],q[6];
rz(1.0300703824081114) q[6];
cx q[16],q[6];
cx q[7],q[14];
rz(1.0470857158853453) q[14];
cx q[7],q[14];
cx q[15],q[7];
rz(0.9679232190394111) q[7];
cx q[15],q[7];
cx q[8],q[11];
rz(1.1745101051118456) q[11];
cx q[8],q[11];
cx q[16],q[8];
rz(1.0992308150942387) q[8];
cx q[16],q[8];
cx q[15],q[9];
rz(1.0874381050448367) q[9];
cx q[15],q[9];
cx q[13],q[10];
rz(0.943079454514871) q[10];
cx q[13],q[10];
cx q[17],q[11];
rz(1.1002434891616344) q[11];
cx q[17],q[11];
cx q[12],q[16];
rz(1.202382988641519) q[16];
cx q[12],q[16];
cx q[17],q[12];
rz(1.1133380064176956) q[12];
cx q[17],q[12];
rx(5.47026242307378) q[0];
rx(5.47026242307378) q[1];
rx(5.47026242307378) q[2];
rx(5.47026242307378) q[3];
rx(5.47026242307378) q[4];
rx(5.47026242307378) q[5];
rx(5.47026242307378) q[6];
rx(5.47026242307378) q[7];
rx(5.47026242307378) q[8];
rx(5.47026242307378) q[9];
rx(5.47026242307378) q[10];
rx(5.47026242307378) q[11];
rx(5.47026242307378) q[12];
rx(5.47026242307378) q[13];
rx(5.47026242307378) q[14];
rx(5.47026242307378) q[15];
rx(5.47026242307378) q[16];
rx(5.47026242307378) q[17];
