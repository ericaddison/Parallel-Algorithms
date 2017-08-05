proctype ABP_Sender(chan request, response, output, input)
{
  bool alt_bit = 0; /* alternating bit */
  bool ack; /* expected acknowledgment */
  do
    :: request?0;
    output!alt_bit;
    do
      :: input?ack;
      if
        :: ack == alt_bit /* correct acknowledgment received */
        -> response!0
        -> alt_bit = !alt_bit
        -> break
        :: ack != alt_bit /* incorrect acknowledgment received,
        consume it and do nothing */
        fi
        :: timeout -> output!alt_bit /* retransmit */
        od
        od
}
proctype ABP_Receiver(chan delivery, input, output)
{
  bool alt_bit = 0; /* alternating bit */
  bool seq = 0; /* received sequence number */
  do
    :: input?seq;
    if
      :: seq == alt_bit /* correct seq. number received */
      -> output!alt_bit /* send acknowledgment */
      -> delivery!0 /* deliver to receiving client */
      -> alt_bit = !alt_bit
      :: seq != alt_bit /* incorrect seq number received */ /* incorrect seq. number received */
      -> output!seq /* resend ack */
      fi
      od
}
