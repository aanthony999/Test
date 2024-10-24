//*--------------------------------------------------------------------
//* THE FOLLOWING EXAMPLE IS CODED IN COBOL:
//*--------------------------------------------------------------------

       IDENTIFICATION DIVISION.
      *****************************************************************
      * MULTIPLY ARRAY A TIMES ARRAY B GIVING ARRAY C                 *
      * USE THE REFERENCE PATTERN CALLABLE SERVICES TO IMPROVE THE    *
      * PERFORMANCE.                                                  *
      *****************************************************************

       PROGRAM-ID. TESTCOB.
       ENVIRONMENT DIVISION.
       DATA DIVISION.
       WORKING-STORAGE SECTION.

      * COPY THE INCLUDE FILE (WHICH DEFINES CSRFORWARD, CSRBACKWARD)
       COPY CSRBPCOB.

      * DIMENSIONS OF ARRAYS - A IS M BY N, B IS N BY P, C IS M BY P
       1   M PIC 9(9) COMP VALUE 200.
       1   N PIC 9(9) COMP VALUE 200.
       1   P PIC 9(9) COMP VALUE 200.

      * ARRAY DECLARATIONS FOR ARRAY A - M = 200, N = 200
       1   A1.
        2   A2 OCCURS 200 TIMES.
         3   A3 OCCURS 200 TIMES.
          4   ARRAY-A PIC S9(8).

      * ARRAY DECLARATIONS FOR ARRAY B - N = 200, P = 200
       1   B1.
        2   B2 OCCURS 200 TIMES.
         3   B3 OCCURS 200 TIMES.
          4   ARRAY-B PIC S9(8).

      * ARRAY DECLARATIONS FOR ARRAY C - M = 200, P = 200
       1   C1.
        2   C2 OCCURS 200 TIMES.
         3   C3 OCCURS 200 TIMES.
          4   ARRAY-C PIC S9(8).

       1   I PIC 9(9) COMP.
       1   J PIC 9(9) COMP.
       1   K PIC 9(9) COMP.
       1   X PIC 9(9) COMP.
       1   ARRAY-A-SIZE PIC 9(9) COMP.
       1   ARRAY-B-SIZE PIC 9(9) COMP.
       1   UNITSIZE PIC 9(9) COMP.
       1   GAP PIC 9(9) COMP.
       1   UNITS PIC 9(9) COMP.
       1   RETCODE PIC 9(9) COMP.
       1   RSNCODE PIC 9(9) COMP.
       PROCEDURE DIVISION.
           DISPLAY " BPAGE PROGRAM START "

      * CALCULATE CSRIRP PARAMETERS FOR INITIALIZING ARRAY A
      * UNITSIZE WILL BE THE SIZE OF ONE ROW.
      * UNITS WILL BE 25
      * SO WE'RE ASKING FOR 25 ROWS TO COME IN AT A TIME
           COMPUTE ARRAY-A-SIZE = M * N * 4
           COMPUTE UNITSIZE = N * 4
           COMPUTE GAP = 0
           COMPUTE UNITS = 25

           CALL "CSRIRP" USING
               ARRAY-A(1, 1),
               ARRAY-A-SIZE,
               CSRFORWARD,
               UNITSIZE,
               GAP,
               UNITS,
               RETCODE,
               RSNCODE

           DISPLAY "FIRST RETURN CODE IS "
           DISPLAY RETCODE

      * CALCULATE CSRIRP PARAMETERS FOR INITIALIZING ARRAY B
      * UNITSIZE WILL BE THE SIZE OF ONE ROW.
      * UNITS WILL BE 25
      * SO WE'RE ASKING FOR 25 ROWS TO COME IN AT A TIME

           COMPUTE ARRAY-B-SIZE = N * P * 4
           COMPUTE UNITSIZE = P * 4
           COMPUTE GAP = 0
           COMPUTE UNITS = 25
           CALL "CSRIRP" USING

               ARRAY-B(1, 1),
               ARRAY-B-SIZE,
               CSRFORWARD,
               UNITSIZE,
               GAP,
               UNITS,
               RETCODE,
               RSNCODE

           DISPLAY "SECOND RETURN CODE IS "
           DISPLAY RETCODE

      * INITIALIZE EACH ARRAY A ELEMENT TO THE SUM OF ITS INDICES
           PERFORM VARYING I FROM 1 BY 1 UNTIL I = M
             PERFORM VARYING J FROM 1 BY 1 UNTIL J = N
               COMPUTE X = I + J
               MOVE X TO ARRAY-A(I, J)
               END-PERFORM
             END-PERFORM

      * INITIALIZE EACH ARRAY B ELEMENT TO THE SUM OF ITS INDICES
           PERFORM VARYING I FROM 1 BY 1 UNTIL I = N
             PERFORM VARYING J FROM 1 BY 1 UNTIL J = P
               COMPUTE X = I + J
               MOVE X TO ARRAY-B(I, J)
             END-PERFORM
           END-PERFORM

      * REMOVE THE REFERENCE PATTERN ESTABLISHED FOR ARRAY A
           CALL "CSRRRP" USING
               ARRAY-A(1, 1),
               ARRAY-A-SIZE,
               RETCODE,
               RSNCODE

           DISPLAY "THIRD RETURN CODE IS "
           DISPLAY RETCODE

      * REMOVE THE REFERENCE PATTERN ESTABLISHED FOR ARRAY B
           CALL "CSRRRP" USING
               ARRAY-B(1, 1),
               ARRAY-B-SIZE,
               RETCODE,
               RSNCODE

           DISPLAY "FOURTH RETURN CODE IS "
           DISPLAY RETCODE

      * CALCULATE CSRIRP PARAMETERS FOR ARRAY A
      * UNITSIZE WILL BE THE SIZE OF ONE ROW.
      * UNITS WILL BE 20
      * SO WE'RE ASKING FOR 20 ROWS TO COME IN AT A TIME
           COMPUTE ARRAY-A-SIZE = M * N * 4
           COMPUTE UNITSIZE = N * 4
           COMPUTE GAP = 0
           COMPUTE UNITS = 20

           CALL "CSRIRP" USING
               ARRAY-A(1, 1),
               ARRAY-A-SIZE,
               CSRFORWARD,
               UNITSIZE,
               GAP,
               UNITS,
               RETCODE,
               RSNCODE

           DISPLAY "FIFTH RETURN CODE IS "
           DISPLAY RETCODE

      * CALCULATE CSRIRP PARAMETERS FOR ARRAY B
      * UNITSIZE WILL BE THE SIZE OF ONE ELEMENT.
      * GAP WILL BE (N-1)*4 (IE. THE REST OF THE ROW).
      * UNITS WILL BE 50
      * SO WE'RE ASKING FOR 50 ELEMENTS OF A COLUMN TO COME IN
      * AT ONE TIME
           COMPUTE ARRAY-B-SIZE = N * P * 4
           COMPUTE UNITSIZE = 4
           COMPUTE GAP = (N - 1) * 4
           COMPUTE UNITS = 50

           CALL "CSRIRP" USING
               ARRAY-B(1, 1),
               ARRAY-B-SIZE,
               CSRFORWARD,
               UNITSIZE,
               GAP,
               UNITS,
               RETCODE,
               RSNCODE

           DISPLAY "SIXTH RETURN CODE IS "
           DISPLAY RETCODE

      * MULTIPLY ARRAY A TIMES ARRAY B GIVING ARRAY C
           PERFORM VARYING I FROM 1 BY 1 UNTIL I = M
             PERFORM VARYING J FROM 1 BY 1 UNTIL J = P
               COMPUTE ARRAY-C(I, J) = 0
               PERFORM VARYING K FROM 1 BY 1 UNTIL K = N
               COMPUTE X = ARRAY-C(I, J) +
                       ARRAY-A(I, K) * ARRAY-B(K, J)
               END-PERFORM
             END-PERFORM
           END-PERFORM

      * REMOVE THE REFERENCE PATTERN ESTABLISHED FOR ARRAY A
           CALL "CSRRRP" USING
               ARRAY-A(1, 1),
               ARRAY-A-SIZE,
               RETCODE,
               RSNCODE

           DISPLAY "SEVENTH RETURN CODE IS "
           DISPLAY RETCODE

      * REMOVE THE REFERENCE PATTERN ESTABLISHED FOR ARRAY B
           CALL "CSRRRP" USING
               ARRAY-B(1, 1),
               ARRAY-B-SIZE,
               RETCODE,
               RSNCODE

           DISPLAY "EIGHTH RETURN CODE IS "
           DISPLAY RETCODE

           DISPLAY " BPAGE PROGRAM END "
           GOBACK.
//*--------------------------------------------------------------------
//* JCL USED TO COMPILE, LINK, THE COBOL PROGRAM
//*--------------------------------------------------------------------
//FCHANGC JOB 'D3113P,D31,?','FCHANG6-6756',CLASS=T,
//     MSGCLASS=H,NOTIFY=FCHANG,REGION=0K
//CCSTEP EXEC EDCCO,
//  CPARM='LIST,XREF,OPTIMIZE,RENT,SOURCE',
//  INFILE='FCHANG.PUB.TEST(C)'
//COMPILE.SYSLIN DD DSN='FCHANG.MPS.OBJ(C),DISP=SHR'
//COMPILE.USERLIB DD  DSN='FCHANG.DECLARE.SET,DISP=SHR
//LKSTEP EXEC EDCPLO,
//    LPARM='AMOD=31,LIST,REFR,RENT,RMOD=ANY,XREF'                      00022007
//PLKED.SYSIN DD DSN='FCHANG.MPS.OBJ(C),DISP=SHR'
//LKED.SYSLMOD DD DSN=RSMID.FBB4417.LINKLIB,DISP=SHR,
//     UNIT=3380,VOL=SER=RSMPAK
//LKED.SYSIN DD *
  LIBRARY IN(CSRIRP,CSRRRP)
  NAME BPGC(R)
//LKED.IN  DD DSN=FCHANG.MPS.OBJ,DISP=SHR
//*--------------------------------------------------------------------
//* LINK PROGRAM
//*--------------------------------------------------------------------
//COBOLLK JOB                                                           00010002
//LINKEDIT EXEC PGM=IEWL,                                               00040000
// PARM='MAP,XREF,LIST,LET,AC=1,SIZE=(1000K,100K)'                      00050000
//SYSLIN   DD DDNAME=SYSIN                                              00051000
//SYSLMOD  DD DSN=REFPAT.USER.LOAD,DISP=OLD                             00052002
//SYSLIB   DD DSN=CEE.SCEELKED,DISP=SHR                                 00053000
//MYLIB    DD DSN=REFPAT.COBOL.OBJ,DISP=SHR                             00053102
//CSRLIB   DD DSN=SYS1.CSSLIB,DISP=SHR                                  00053202
//SYSPRINT DD SYSOUT=H                                                  00053300
//*                                                                     00053400
//SYSUT1   DD UNIT=SYSDA,SPACE=(TRK,(20,10))                            00053500
//SYSUT2   DD UNIT=SYSDA,SPACE=(TRK,(20,10))                            00053600
//SYSIN    DD *                                                         00053700
  INCLUDE MYLIB(COBOL)                                                  00053802
  LIBRARY CSRLIB(CSRIRP,CSRRRP)                                         00053901
  NAME COBLOAD(R)                                                       00054002
/*                                                                      00055000
//*--------------------------------------------------------------------
//* JCL USED TO EXECUTE THE COBOL PROGRAM
//*--------------------------------------------------------------------
//COB2  JOB  MSGLEVEL=(1,1),TIME=1440                                   00010000
//GO     EXEC  PGM=COBLOAD                                              00020001
//STEPLIB  DD  DSNAME=CEE.SCEERUN,DISP=SHR                              00030001
//         DD  DSN=REFPAT.USER.LOAD,DISP=SHR,VOL=SER=RSMPAK,            00040001
//     UNIT=3380                                                        00041001
//SYSABOUT DD  SYSOUT=*                                                 00050000
//SYSOUT   DD  SYSOUT=A                                                 00051001
//SYSDBOUT DD  SYSOUT=*                                                 00060000
//SYSUDUMP DD  SYSOUT=*                                                 00070000
