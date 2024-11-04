*
* COBOL COMPILER SAMPLE
* Example 5 - 
* 
* by surender, www.suren.space
* 
PRG5 Write  a program to accept the date and time from system and display in the given format( USING MOVE CORRESPONDING and FILLER)  DD/MM/YYYY & HH:MM:SS.    
  
        IDENTIFICATION DIVISION.                   
        PROGRAM-ID. PRG5.                          
        ENVIRONMENT DIVISION.                      
        DATA DIVISION.                             
        WORKING-STORAGE SECTION.                   
        01 G1.                                     
           02 YY PIC X(2).                         
           02 MM PIC X(2).                         
           02 DD PIC X(2).                         
        01 G2.                                     
           02 DD PIC X(2).                         
           02 FILLER PIC X VALUE '-'.              
           02 MM PIC X(2).                  
           02 FILLER PIC X VALUE ‘-‘.     
           02 YY PIC X(2).                     
        01 G3.                                 
           02 HH PIC X(2).                     
           02 MM PIC X(2).                     
           02 SS PIC X(2).                     
        01 G4.                                 
           02 HH PIC X(2).                     
           02 FILLER PIC X VALUE ':'.          
           02 MM PIC X(2).                     
           02 FILLER PIC X VALUE ':'.          
           02 SS PIC X(2).                     
        PROCEDURE DIVISION.                    
               ACCEPT G1 FROM DATE.            
               ACCEPT G3 FROM TIME.            
               MOVE CORRESPONDING G1 TO G2.    
               MOVE CORRESPONDING G3 TO G4.    
               DISPLAY G2.                   
               DISPLAY G4.                   
               STOP RUN.
