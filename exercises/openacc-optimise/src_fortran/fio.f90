!
!  This file contains Fortran subroutines for the MPI Casestudy.
!
!  "datread" reads in an edges data file and can be called as follows:
!
!     real buf(M, N)
!     call datread('edge.dat', buf, M, N)
!
!  "pgmwrite" writes an image as a PGM file, and can be called as follows:
!
!     real buf(M, N)
!     call pgmwrite('image.pgm', buf, M, N)
!


!  Routine to read an "edges" data file into a 2D floating point array
!  x(nx,ny), using unit 12 for IO.


subroutine datread(filename, x, nx, ny)

  implicit none

  character*(*) :: filename
  integer :: nx, ny, nxt, nyt
  real, dimension(nx, ny) :: x

  integer i, j

  integer, parameter :: iounit = 12

  open(unit=iounit, file=filename)

  read(iounit,*) nxt, nyt

  if (nx .ne. nxt .or. ny .ne. nyt) then
    write(*,*) 'datread: size mismatch, (nx,ny) = (', nxt, ',', nyt, &
               ') expected (', nx, ',', ny, ')'
    stop
  end if

  read(iounit,*) ((x(i,ny-j+1), i=1,nx), j=1,ny)

  close(unit=iounit)

end subroutine datread


!  Routine to write a PGM image file from a 2D floating point array
!  x(nx,ny). Uses unit 10 for IO.

subroutine pgmwrite(filename, x, nx, ny)

  implicit none

  character*(*) :: filename
  integer :: nx, ny

  real,    dimension(nx, ny) :: x

  real,    dimension(nx, ny) :: tmp
  integer, dimension(nx, ny) :: grey

  real :: tmin, tmax
  real, parameter :: thresh = 255.0

  integer, parameter :: iounit = 10

  integer :: i, j

  tmp(:,:) = x(:,:)

!  Find the max and min absolute values of the array

  tmin = minval(abs(tmp(:,:)))
  tmax = maxval(abs(tmp(:,:)))

!  Scale the values appropriately so the lies between 0 and thresh

  if (tmin .lt. 0 .or. tmax .gt. thresh) then

    tmp(:,:) = int((thresh*((abs(tmp(:,:)-tmin))/(tmax-tmin))) + 0.5)

  else

    tmp(:,:) = int(abs(tmp(:,:)) + 0.5)

  end if

!  Increase the contrast by boosting the lower values

  grey(:,:) = thresh * sqrt(tmp(:,:)/thresh)

  open(unit=iounit, file=filename)

  write(iounit,fmt='(''P2''/''# Written by pgmwrite'')')
  write(iounit,*) nx, ny
  write(iounit,*) int(thresh)
  write(iounit,fmt='(16(i3,'' ''))') ((grey(i,ny-j+1), i=1,nx), j=1,ny)  

  close(unit=iounit)

end subroutine pgmwrite
